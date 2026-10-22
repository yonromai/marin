"""Finite TIM measurements. Only installed in the experiment worktree."""

import asyncio

import cProfile

from dataclasses import replace

import hashlib

import importlib.metadata

import json

import math

import time

import io

import pstats

import resource

from pathlib import Path

import numpy as np

import psutil

import fsspec

import ray

import ray.cloudpickle

import torch

from omegaconf import OmegaConf

from rigging.filesystem.s3_compat import configure_coreweave_s3

from skyrl_train.entrypoints.main_base import BasePPOExp

from skyrl_train.objective.correction import compute_correction

from skyrl_train.config.objective_spec import off_policy_correction

from skyrl_train.objective.score_centering import collate_score_centering

from skyrl_train.trainer import RayPPOTrainer, convert_prompts_responses_to_batch_tensors, _token_staleness_metrics

from skyrl_train.training_batch import TrainingInputBatch

from skyrl_train.timing_observability import StepWallTime

from skyrl_train.rollouts.context import TrainingContext, RolloutRequestSpec





def host_memory():
    return {
        "rss_current": psutil.Process().memory_info().rss,
        "rss_cumulative_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected report value: {type(value).__name__}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value).encode()).hexdigest()


def profile_summary(profile):
    output = io.StringIO()
    stats = pstats.Stats(profile, stream=output).sort_stats("cumulative")
    stats.print_stats(25)
    selected = {}
    for (_, _, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        if name in (
            "align_student_topk", "_student_topk", "training_batch", "_create_chat_logprobs",
            "write_rollout", "_put", "dump", "dumps", "_pickle",
            "_response_logprobs",
            "ppo_tis_score_centering_correction",
            "convert_ids_list_to_tokens",
            "_update_sample_logprobs",
            "_postprocess_outputs",
            "collate_score_centering",
            "_create_chat_logprobs",
            "_get_top_logprobs",
            "select_flat_response_topk",
            "select_flat_chat_response_topk",
        ):
            row = selected.setdefault(name, {"calls": 0, "own_seconds": 0, "cumulative_seconds": 0})
            row["calls"] += calls
            row["own_seconds"] += own
            row["cumulative_seconds"] += cumulative
    return {"selected": selected, "top": output.getvalue()}


def profile_rollout_worker(worker):
    """Profile the first ordinary and head request per actor, within excluded warmups."""
    original = worker.run_task
    seen = set()

    async def run_task(task, writer):
        heads = bool((task.request.sampling_params or {}).get("logprobs"))
        if heads in seen:
            return await original(task, writer)
        seen.add(heads)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            return await original(task, writer)
        finally:
            profiler.disable()
            print("TIM_ROLLOUT_PROFILE " + json.dumps({
                "heads": heads,
                "scope": "first request per actor including serialization/write; concurrent actor tasks can interleave",
                **profile_summary(profiler),
            }), flush=True)

    worker.run_task = run_task
    return worker


class FixedPromptDataset:
    def __init__(self, dataset, size, indices=None):
        self.dataset, self.size = dataset, size
        self.indices = list(range(size)) if indices is None else indices
        assert len(self.indices) == size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def uid(self, index):
        return self.dataset.uid(self.indices[index])

    def collate_fn(self, rows):
        return self.dataset.collate_fn(rows)


class BenchmarkRequestSpec:
    def __init__(self, original, limits, flat, interval):
        self.original, self.limits, self.flat, self.probe_interval = original, limits, flat, interval

    def request(self, prompt, policy_step, *, probe=False):
        request = (
            self.original.request(prompt, policy_step, probe=True)
            if probe
            else self.original.request(prompt, policy_step)
        )
        limit, seed = self.limits[prompt["uid"]]
        return replace(
            request,
            sampling_params={
                **request.sampling_params,
                "max_tokens": limit,
                "ignore_eos": True,
                "seed": seed,
                "flat_logprobs": self.flat and (request.sampling_params.get("logprobs") or 0) > 0,
            },
        )


def checked_response_logprobs(original, stats):
    def checked(
        logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids=None, candidate_positions=None
    ):
        result = (
            original(logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids)
            if candidate_positions is None
            else original(
                logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids, candidate_positions
            )
        )
        if candidate_ids is None:
            return result
        for row in range(len(sequences)):
            valid = candidate_ids[row].ge(0).any(-1).nonzero().flatten().tolist()
            if not valid:
                continue
            prompt_length = int(attention_mask[row, :-num_actions].sum())
            for index in sorted({valid[0], valid[len(valid) // 2], valid[-1]}):
                position = index if candidate_positions is None else int(candidate_positions[row, index])
                # Independent causal-position calculation on the real compacted model logits.
                normalized = logits[row, prompt_length - 1 + position].float().log_softmax(-1)
                token = sequences[row, -num_actions + position]
                ids = candidate_ids[row, index].long()
                chosen_error = abs(float(normalized[token] - result.scores[row, position]))
                head_error = float((normalized[ids] - result.selected_logprobs[row, index]).abs().max())
                stats["positions"] += 1
                stats["chosen_max_error"] = max(stats["chosen_max_error"], chosen_error)
                stats["head_max_error"] = max(stats["head_max_error"], head_error)
        return result

    return checked


def reference_forward(worker, data):
    original = worker.model._response_logprobs
    stats = {"positions": 0, "chosen_max_error": 0.0, "head_max_error": 0.0}
    worker.model._response_logprobs = checked_response_logprobs(original, stats)
    try:
        output = worker.forward(data)
    finally:
        worker.model._response_logprobs = original
    output.metadata = {**(output.metadata or {}), "performance_reference": stats}
    return output


class PerformanceExp(BasePPOExp):
    def get_train_dataset(self):
        dataset = super().get_train_dataset()
        spec = self.cfg.trainer.score_centering_performance
        if spec.source_rows is None:
            return dataset
        assert len(self.cfg.data.train_data) == 1
        assert hashlib.sha256(Path(self.cfg.data.train_data[0]).read_bytes()).hexdigest() == spec.source_data_sha256
        positions = {int(row): index for index, row in enumerate(dataset.dataframe["campaign_source_row"])}
        indices = [positions[int(row)] for row in spec.source_rows]
        return FixedPromptDataset(dataset, len(indices), indices)

    def get_trainer(self, **kwargs):
        trainer = super().get_trainer(**kwargs)
        trainer.__class__ = PerformanceTrainer
        return trainer


def memory_stats(reset=False):
    # Allocator reads only. No device synchronization or background polling.
    return {"device": torch.cuda.get_device_name(), "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(), "host": host_memory()}

class PerformanceTrainer(RayPPOTrainer):
    def save_probe(self):
        self.probe["applied_updates"] = self._performance_applied_updates
        path = Path(self.cfg.trainer.export_path) / "tim-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.probe, indent=2, allow_nan=False, default=json_value) + "\n")
        with fsspec.open(self.cfg.trainer.score_centering_performance.output_uri, "wt") as output:
            json.dump(self.probe, output, allow_nan=False, default=json_value)
        print(
            "READINESS "
            + json.dumps(
                {
                    "complete": self.probe.get("complete", False),
                    "windows": len(self.probe["windows"]),
                    "qualification": list(self.probe["qualification"]),
                }
            ),
            flush=True,
        )


    async def settings(self, width, enabled, telemetry=True, diagnostic_tokens=0, temperature=1.0):
        self.cfg.trainer.algorithm.score_centering_topk = width
        self.cfg.trainer.algorithm.score_centering_enabled = enabled
        self.cfg.trainer.algorithm.score_centering_diagnostic_tokens = diagnostic_tokens
        self.cfg.generator.sampling_params.logprobs = width
        self.cfg.generator.sampling_params.temperature = temperature
        self._training_metrics_enabled = telemetry
        self._performance_inference_sampling = await asyncio.gather(
            *(
                engine.inference_engine_actor.performance_sampling.remote(temperature)
                for engine in self.inference_engine_client.engines
            )
        )
        return await asyncio.to_thread(
            ray.get,
            self.policy_model.async_run_ray_method(
                "pass_through", "performance_settings", width, enabled, diagnostic_tokens, temperature
            ),
        )


    async def inference_memory(self, reset=False):
        return await asyncio.gather(
            *(
                engine.inference_engine_actor.performance_memory.remote(reset)
                for engine in self.inference_engine_client.engines
            )
        )


    async def generate_workload(self, width, *, temperature=1.0, short=False, flat=None):
        limits = [8, 24, 64, 128] if short else [64, 256, 1024, 4096]
        outputs = []
        for group, limit in enumerate(limits):
            prompts = self.prompts[group::4]
            result = await self.inference_engine_client.generate(
                {
                    "prompt_token_ids": prompts,
                    "sampling_params": {
                        "temperature": temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "max_tokens": limit,
                        "ignore_eos": True,
                        "seed": 611 + group,
                        "logprobs": width,
                        "flat_logprobs": self.cfg.trainer.score_centering_performance.variant == "candidate"
                        if flat is None
                        else flat,
                    },
                }
            )
            assert all(len(row) == limit for row in result["response_ids"])
            assert all(set(row) == {1} for row in result["response_policy_steps"])
            outputs.extend((prompt, result, index) for index, prompt in enumerate(prompts))
        return outputs


    def batch(self, outputs, width):
        prompts = [prompt for prompt, _, _ in outputs]
        responses = [output["response_ids"][index] for _, output, index in outputs]
        chosen = [np.array(output["response_logprobs"][index]) for _, output, index in outputs]
        masks = [[int(position % 31 != 7) for position in range(len(row))] for row in responses]
        masks[0] = [0] * len(masks[0])
        rewards = [[0.0] * len(row) for row in responses]
        values = convert_prompts_responses_to_batch_tensors(
            self.tokenizer, prompts, responses, rewards, masks, chosen, None, None
        )
        sequences, attention, response_mask, _, loss_mask, sampled, _, _ = values
        policy_steps = torch.full_like(loss_mask, -1, dtype=torch.int32)
        for row, (_, output, index) in enumerate(outputs):
            stamps = output["response_policy_steps"][index]
            policy_steps[row, : len(stamps)] = torch.tensor(stamps, dtype=torch.int32)
        staleness = torch.where(response_mask.bool(), self.global_step - policy_steps, 0)
        payload = {
            "sequences": sequences,
            "attention_mask": attention,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_logprobs": sampled,
            "rollout_staleness": staleness,
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros_like(loss_mask),
            # Fixed signed advantages qualify optimizer execution, without a learning-benefit experiment.
            "advantages": loss_mask * torch.tensor([1.0, -1.0] * (len(prompts) // 2)).unsqueeze(-1),
        }
        if width:
            trajectory = {
                "loss_masks": masks,
                "student_topk_indices": [np.asarray(output["student_topk_indices"][i]) for _, output, i in outputs],
                "behavior_topk_logprobs": [np.asarray(output["behavior_topk_logprobs"][i]) for _, output, i in outputs],
            }
            ids, behavior = collate_score_centering(
                trajectory, responses, response_mask, width, sampled_logprobs=sampled
            )
            payload.update(score_topk_indices=ids, score_behavior_logprobs=behavior)
        batch = TrainingInputBatch(payload)
        if self._training_metrics_enabled:
            self.all_metrics.update(_token_staleness_metrics(staleness, loss_mask, policy_steps))
        batch.metadata = {"response_length": response_mask.shape[1], "uids": list(map(str, range(len(prompts))))}
        return batch


    async def reference(self, batch):
        keys = ["sequences", "attention_mask", "response_mask", "score_topk_indices"]
        data = batch.select(keys, metadata_keys=["response_length"])
        rows = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("mesh", "performance_reference", data)
        )
        stats = [
            row.metadata["performance_reference"] for row in rows if row.metadata["performance_reference"]["positions"]
        ]
        assert stats and max(row["chosen_max_error"] for row in stats) < 2e-5
        assert max(row["head_max_error"] for row in stats) < 2e-5
        return stats


    async def prefix_replay(self, outputs, *, temperature=1.0):
        prefixes, overrides, expected, samples = [], [], [], []
        for prompt, output, index in outputs[:: max(1, len(outputs) // 8)]:
            response = output["response_ids"][index]
            for position in sorted({0, len(response) // 2, len(response) - 1}):
                ids = np.asarray(output["student_topk_indices"][index][position]).tolist()
                scores = np.asarray(output["behavior_topk_logprobs"][index][position]).tolist()
                prefixes.append(prompt + response[:position])
                overrides.append({"logprob_token_ids": ids})
                expected.append(scores)
                samples.append(
                    {
                        "prompt_ids": prompt,
                        "response_prefix_ids": response[:position],
                        "chosen_token_id": response[position],
                        "position": position,
                        "captured_support_ids": ids,
                        "captured_support_logprobs": scores,
                        "captured_chosen_logprob": output["response_logprobs"][index][position],
                        "temperature": temperature,
                        "generating_policy_step": output["response_policy_steps"][index][position],
                    }
                )
        replay = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        shifted = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": [prefix[:-1] for prefix in prefixes],
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        chosen_replay = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 1,
                },
                "sampling_params_per_prompt": [
                    {"logprob_token_ids": [sample["chosen_token_id"]]} for sample in samples
                ],
            }
        )
        gaps = []
        shifted_gaps = []
        for index, target in enumerate(expected):
            np.testing.assert_array_equal(
                replay["student_topk_indices"][index][0], overrides[index]["logprob_token_ids"]
            )
            gaps.extend(np.asarray(replay["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            shifted_gaps.extend(np.asarray(shifted["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            samples[index]["prefix_replay_support_logprobs"] = replay["behavior_topk_logprobs"][index][0]
            samples[index]["prefix_replay_chosen_logprob"] = float(chosen_replay["behavior_topk_logprobs"][index][0][0])
        assert np.abs(gaps).mean() < np.abs(shifted_gaps).mean()
        return {
            "prefixes": len(prefixes),
            "prefix_ids_sha256": digest(prefixes),
            "support_ids_sha256": digest(overrides),
            "temperature": temperature,
            "same_engine_decode_prefill_gap_abs_mean": float(np.abs(gaps).mean()),
            "same_engine_decode_prefill_gap_abs_max": float(np.abs(gaps).max()),
            "one_token_shift_control_gap_abs_mean": float(np.abs(shifted_gaps).mean()),
            "policy_steps": sorted({step for row in replay["response_policy_steps"] for step in row}),
            "samples": samples,
        }


    async def update(self, batch):
        started = time.perf_counter()
        batch = await asyncio.to_thread(self.fwd_logprobs_values_reward, batch)
        forward = time.perf_counter() - started
        correction = compute_correction(
            batch["action_log_probs"],
            batch["rollout_logprobs"],
            batch["loss_mask"],
            off_policy_correction(self.cfg.trainer.algorithm),
        )
        batch["correction_weights"] = correction.weights
        if "score_probe_positions" in batch:
            self._finish_score_probe(batch)
        trained = time.perf_counter()
        status = await asyncio.to_thread(self.train_critic_and_policy, batch)
        update = time.perf_counter() - trained
        trainer_seconds = time.perf_counter() - started
        assert status["policy_update_steps"] == 1 and status["skipped_steps"] == 0
        assert math.isfinite(status["raw_grad_norm"]) and status["raw_grad_norm"] > 0
        self._performance_applied_updates += int(status["policy_update_steps"])
        self._policy_version_step = self.global_step
        await asyncio.to_thread(self._offload_policy_optimizer, self.all_timings, timer_label="probe_offload")
        memory = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("pass_through", "performance_memory")
        )
        return {
            "trainer_seconds": trainer_seconds,
            "forward_seconds": forward,
            "update_seconds": update,
            "status": status,
            "learner_memory": memory,
            "metrics": dict(self.all_metrics),
            "timings": dict(self.all_timings),
        }


    def postprocess_trajectory_batch(self, trajectory_batch, uids):
        result = super().postprocess_trajectory_batch(trajectory_batch, uids)
        if "score_probe_positions" in result:
            self._performance_probe_rows = [
                {
                    "uid": uids[row],
                    "response_tokens": len(result["response_ids"][row]),
                    "exported_positions_before_trainer_masks": len(positions),
                }
                for row, positions in enumerate(result["score_probe_positions"])
                if positions is not None
            ]
        if self._performance_fixed_advantages:
            masked_row = uids.index(self._performance_masked_uid)
            result["loss_masks"] = [
                [int(bool(value) and index % 31 != 7 and row != masked_row) for index, value in enumerate(mask)]
                for row, mask in enumerate(result["loss_masks"])
            ]
        return result


    def finalize_advantages_for_training(self, data):
        result = super().finalize_advantages_for_training(data)
        if self._performance_fixed_advantages:
            signs = torch.tensor([1.0 if index % 2 == 0 else -1.0 for index in range(len(data["loss_mask"]))])
            result["advantages"] = data["loss_mask"] * signs.unsqueeze(-1)
        return result


    def _finish_score_probe(self, data):
        if self._performance_record_evidence:
            positions = data["score_probe_positions"]
            valid = (data["score_topk_indices"] >= 0).all(-1) & data["loss_mask"].gather(1, positions).bool()
            self.probe.setdefault("probe_evidence", []).append(
                {
                    "phase": self._performance_phase,
                    "training_step": self.global_step,
                    "synthetic_advantages": self._performance_fixed_advantages,
                    "positions": positions[valid].numpy(),
                    "current_old_head": data["score_old_logprobs"][valid].numpy(),
                    "behavior_head": data["score_behavior_logprobs"][valid].numpy(),
                    "current_old_sampled": data["action_log_probs"].gather(1, positions)[valid].numpy(),
                    "behavior_sampled": data["rollout_logprobs"].gather(1, positions)[valid].numpy(),
                    "advantages": data["advantages"].gather(1, positions)[valid].numpy(),
                    "source_steps": data["score_probe_source_steps"][valid].numpy(),
                    "loss_tokens": int(data["loss_mask"].bool().sum()),
                }
            )
        super()._finish_score_probe(data)


    async def frontend_profile(self, start):
        method = "performance_profile_start" if start else "performance_profile_stop"
        return await asyncio.gather(
            *(
                getattr(engine.inference_engine_actor, method).remote()
                for engine in self.inference_engine_client.engines
            )
        )


    async def learner_profile(self, start):
        return await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("pass_through", "performance_profile", start)
        )


    async def native_window(self, phase, *, asynchronous=False):
        self._performance_phase = phase
        self._performance_probe_rows = []
        self.all_metrics, self.all_timings = {}, {}
        started = time.perf_counter()
        await self._publish(self.global_step + 1)
        self.global_step += 1
        wall = StepWallTime(OmegaConf.to_container(self.cfg.trainer.step_phase_budgets))
        epoch = (self.global_step - 1) // self.num_steps_per_epoch if asynchronous else 0
        data = await self._train_step(epoch, wall)
        core_seconds = time.perf_counter() - started
        self.all_metrics.update(wall.finish(core_seconds))
        payload = {**self.all_metrics, **{f"timing/{key}": value for key, value in self.all_timings.items()}}
        self._log_metrics_stdout(payload, step=self.global_step, kind="train")
        self.tracker.log(payload, step=self.global_step, commit=self.cfg.trainer.tracker_commit_each_step)
        elapsed = time.perf_counter() - started
        assert self.all_metrics["policy/policy_update_steps"] == 1
        assert self.all_metrics["policy/skipped_steps"] == 0
        norm = self.all_metrics["policy/raw_grad_norm"]
        assert math.isfinite(norm) and (norm > 0 if self._performance_fixed_advantages else norm >= 0)
        assert self.all_metrics["async/staleness_tokens_stamped"] == 1.0
        assert self.all_metrics["async/staleness_tokens_max"] <= (
            self.cfg.trainer.rollout_buffer.max_staleness_steps if asynchronous else 0
        )
        self._performance_applied_updates += 1
        result = {
            "phase": phase,
            "total_seconds": elapsed,
            "core_seconds": core_seconds,
            "started_at_monotonic": started,
            "cycle_completed_at_monotonic": started + elapsed,
            "metric_emission_seconds": elapsed - core_seconds,
            "tokens": int(data["response_mask"].sum()),
            "loss_tokens": int(data["loss_mask"].bool().sum()),
            "nonzero_advantage_tokens": int((data["advantages"] != 0).sum()),
            "sequence_shape": list(data["sequences"].shape),
            "prompt_lengths": (data["attention_mask"].sum(-1) - data["response_mask"].sum(-1)).int().tolist(),
            "row_loss_tokens": data["loss_mask"].bool().sum(-1).int().tolist(),
            "optimizer_input_tensor_bytes": {
                name: value.numel() * value.element_size()
                for name, value in data.items()
                if isinstance(value, torch.Tensor)
            },
            "driver_host": host_memory(),
            "response_lengths": data["response_mask"].sum(-1).int().tolist(),
            "sequence_sha256": hashlib.sha256(data["sequences"].numpy().tobytes()).hexdigest(),
            "uids": data.metadata.get("uids"),
            "probe_capture": self._performance_probe_rows,
            "metrics": dict(self.all_metrics),
            "timings": dict(self.all_timings),
            "applied_updates": self._performance_applied_updates,
            "synthetic_advantages": self._performance_fixed_advantages,
            "inference_sampling": getattr(self, "_performance_inference_sampling", None),
            "asynchronous": asynchronous,
        }
        return result


    def finish_performance_report(self):
        self.probe["complete"] = True
        self.probe["applied_updates"] = self._performance_applied_updates
        self.save_probe()


    def convert_to_training_input(self, *args, **kwargs):
        if not self._performance_driver_profile:
            return super().convert_to_training_input(*args, **kwargs)
        profile = cProfile.Profile()
        result = profile.runcall(super().convert_to_training_input, *args, **kwargs)
        self._performance_driver_profiles.append({"phase": "collation", **profile_summary(profile)})
        return result


    def fwd_logprobs_values_reward(self, *args, **kwargs):
        if not self._performance_driver_profile:
            return super().fwd_logprobs_values_reward(*args, **kwargs)
        profile = cProfile.Profile()
        result = profile.runcall(super().fwd_logprobs_values_reward, *args, **kwargs)
        self._performance_driver_profiles.append({"phase": "old_forward_and_collection", **profile_summary(profile)})
        return result


    async def transfer_measurement(self, captured):
        selected = [(output, index) for _, output, index in captured if len(output["response_ids"][index]) == 4096][:4]
        engine = self.inference_engine_client.engines[0].inference_engine_actor
        records = {}
        for name, keys in (
            ("chosen", ("response_ids", "response_logprobs", "response_policy_steps")),
            ("head", ("response_ids", "response_logprobs", "response_policy_steps", "student_topk_indices", "behavior_topk_logprobs")),
        ):
            payload = {key: [output[key][index] for output, index in selected] for key in keys}
            elapsed = []
            for repetition in range(7):
                started = time.perf_counter()
                returned = await engine.performance_echo.remote(payload)
                duration = time.perf_counter() - started
                assert returned["response_ids"] == payload["response_ids"]
                if repetition:
                    elapsed.append(duration)
            records[name] = {"ray_actor_roundtrip_seconds": elapsed,
                             "pickle5_bytes": len(ray.cloudpickle.dumps(payload, protocol=5))}
        return {"scope": "driver to inference frontend and back, four real 4096-token numeric responses; no generation, trainer transfer or S3",
                "arms": records}

    def trainer_probe_batch(self, captured):
        batch = self.batch(captured, 128)
        dense_ids = batch.pop("score_topk_indices")
        dense_scores = batch.pop("score_behavior_logprobs")
        rows, _, width = dense_ids.shape
        positions = torch.zeros((rows, 32), dtype=torch.int64)
        ids = torch.full((rows, 32, width), -1, dtype=torch.int32)
        scores = torch.zeros((rows, 32, width), dtype=torch.float32)
        sources = torch.full((rows, 32), -1, dtype=torch.int64)
        long_rows = [index for index, row in enumerate(batch["response_mask"].sum(-1)) if row == 4096][:4]
        assert len(long_rows) == 4
        for row in long_rows:
            valid = batch["loss_mask"][row].nonzero().flatten()
            chosen = valid[torch.linspace(0, len(valid) - 1, 32).long()]
            positions[row] = chosen
            ids[row] = dense_ids[row, chosen]
            scores[row] = dense_scores[row, chosen]
            sources[row] = 1
        batch.update(score_topk_indices=ids, score_behavior_logprobs=scores,
                     score_probe_positions=positions, score_probe_source_steps=sources)
        return batch

    async def configure_context(self, *, width, enabled, interval, fixed=False):
        await self.context.close()
        await self.settings(width, enabled)
        self.cfg.trainer.algorithm.score_centering_probe.interval = interval
        dataset = FixedPromptDataset(self.train_dataset, 32) if fixed else self.train_dataset
        self.context = TrainingContext.from_config(self.cfg, dataset, self.trajectory_runner)
        if fixed:
            limits = [4096, 256, 1024, 64]
            placements = {dataset.uid(index): (limits[index % 4], 1701 + index) for index in range(32)}
            self.context._request_spec = BenchmarkRequestSpec(
                RolloutRequestSpec.from_config(self.cfg), placements, True, interval
            )
            self._performance_masked_uid = dataset.uid(0)
        self.context.start()

    async def controlled_measurements(self):
        spec = self.cfg.trainer.score_centering_performance
        # The longest raw qualification prefixes have exactly 4096 prompt tokens.
        # Native task cycles keep the retained source prompts and force only response lengths.
        newline = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        for index in range(3, len(self.prompts), 4):
            prompt = self.prompts[index]
            assert len(prompt) <= 4096
            self.prompts[index] = [newline] * (4096 - len(prompt)) + prompt
        self.probe["controlled_raw_prompt_sha256"] = digest(self.prompts)
        await self.settings(128, False)
        self.global_step = 1
        captured = await self.generate_workload(128, flat=True)
        self.probe["qualification"]["prefix_replay"] = await self.prefix_replay(captured)
        batch = self.batch(captured, 128)
        assert batch["sequences"].shape[1] == 8192
        self.probe["qualification"]["reference"] = await self.reference(batch)
        self.probe["qualification"]["transfer"] = await self.transfer_measurement(captured)
        self.probe["qualification"]["sequence_shape"] = list(batch["sequences"].shape)
        self.probe["qualification"]["response_lengths"] = [len(output["response_ids"][index]) for _, output, index in captured]
        self.save_probe()
        del batch

        # Full forward + optimizer compute on identical captured tokens. LR is zero,
        # so optimizer state advances but weights stay fixed. Probe summaries are included.
        self.probe["trainer_windows"] = []
        modes = [("ordinary", 0, False), ("sc", 128, True), ("probe", 0, False)]
        for repetition in range(int(spec.repetitions) + 1):
            order = modes[repetition % len(modes):] + modes[:repetition % len(modes)]
            for name, width, enabled in order:
                await self.settings(width, enabled)
                self.all_metrics, self.all_timings = {}, {}
                self.global_step = 1
                batch = self.trainer_probe_batch(captured) if name == "probe" else self.batch(captured, width)
                result = await self.update(batch)
                result.update(mode=name, repetition=repetition, warmup=repetition == 0,
                              tokens=int(batch["response_mask"].sum()),
                              loss_tokens=int(batch["loss_mask"].bool().sum()),
                              sequence_sha256=hashlib.sha256(batch["sequences"].numpy().tobytes()).hexdigest())
                self.probe["trainer_windows"].append(result)
                self.save_probe()
                del batch
        del captured

        self._performance_fixed_advantages = True
        self._performance_record_evidence = False
        self.global_step = 0
        await self._sync_policy_for_rollouts(reason="controlled_native_start")
        self._last_score_probe_time = self._last_score_probe_step = None
        modes = [("ordinary", 0, False), ("capture", 128, False), ("sc", 128, True), ("probe", 0, False)]
        expected = sorted([4096, 256, 1024, 64] * 32)
        self.probe["controlled_native"] = {
            "weights": "original SFT; AdamW lr=min_lr=0; applied updates compute gradients and moments",
            "input_workload": "same first 32 retained prompt groups; seed=1701+group; fixed response lengths, ignore_eos",
            "synthetic_advantages": "alternating +1/-1; every 31st token and the first response masked",
            "scope": "serialized native collection/object store/trainer/sync plus normal metric emission; context recreation outside window",
            "repetitions_after_one_warmup": int(spec.repetitions),
        }
        for repetition in range(int(spec.repetitions) + 1):
            order = modes[repetition % len(modes):] + modes[:repetition % len(modes)]
            for name, width, enabled in order:
                interval = 1 if name == "probe" else 0
                await self.configure_context(width=width, enabled=enabled, interval=interval, fixed=True)
                profile = repetition == 0
                self._performance_phase = f"native/{name}/{repetition}"
                self._performance_record_evidence = name == "probe" and profile
                self._performance_driver_profiles = []
                self._performance_driver_profile = profile
                if profile:
                    await self.frontend_profile(True)
                    await self.learner_profile(True)
                async with asyncio.timeout(int(spec.window_timeout_seconds)):
                    result = await self.native_window(self._performance_phase)
                assert sorted(result["response_lengths"]) == expected
                if name == "probe":
                    assert result["metrics"]["score_centering_probe/observed"] == 1
                    assert result["metrics"]["score_centering_probe/long_responses"] > 0
                if profile:
                    self.probe.setdefault("profiles", {})[name] = {
                        "frontend": await self.frontend_profile(False),
                        "learner": await self.learner_profile(False),
                        "driver": self._performance_driver_profiles,
                    }
                self._performance_driver_profile = False
                self._performance_record_evidence = False
                result.update(mode=name, repetition=repetition, warmup=profile)
                self.probe["windows"].append(result)
                self.save_probe()

        # Measure the proposed default as a real, consumed sequence of observations.
        # Its first group is long; masked response positions cannot supply evidence.
        interval = int(spec.default_interval)
        await self.configure_context(width=0, enabled=False, interval=interval, fixed=True)
        self._last_score_probe_time = self._last_score_probe_step = None
        self.probe["periodic_windows"] = []
        for index in range(2 * interval + 2):
            async with asyncio.timeout(int(spec.window_timeout_seconds)):
                result = await self.native_window(f"periodic/{index}")
            result.update(mode="periodic", repetition=index, warmup=index < 2)
            self.probe["periodic_windows"].append(result)
            self.save_probe()
        assert sum(row["metrics"].get("score_centering_probe/observed", 0) for row in self.probe["periodic_windows"]) >= 3
        await self.context.close()

    async def coverage_measurements(self):
        spec = self.cfg.trainer.score_centering_performance
        self._performance_fixed_advantages = False
        self._performance_record_evidence = False
        self.cfg.trainer.rollout_buffer.max_staleness_steps = 4
        self.cfg.trainer.rollout_buffer.max_in_flight = 64
        self.probe["asynchronous_windows"] = []
        self.probe["asynchronous_controls"] = {
            "max_staleness_steps": 4, "max_in_flight": 64,
            "sampling": "natural EOS; 4096 response / 8192 context budget; retained 192-row order",
            "advantages": "normal GRPO rewards, no injected advantages",
            "interpretation": "coverage and runtime only; sequential changing trajectories are not a causal cost estimate",
        }
        for name, width, enabled, interval, steps in (
            ("probes", 0, False, int(spec.default_interval), int(spec.async_max_steps)),
            ("sc", 128, True, 0, 4),
        ):
            await self.configure_context(width=width, enabled=enabled, interval=interval)
            self._last_score_probe_time = self._last_score_probe_step = None
            observed = 0
            for index in range(steps):
                async with asyncio.timeout(int(spec.window_timeout_seconds)):
                    result = await self.native_window(f"async/{name}/{index}", asynchronous=True)
                result.update(mode=name, repetition=index, warmup=index < 2)
                self.probe["asynchronous_windows"].append(result)
                observed += int(result["metrics"].get("score_centering_probe/observed", 0))
                self.save_probe()
                if name == "probes" and index >= 2 * interval + 1 and observed >= 3:
                    break
            if name == "probes":
                assert observed >= 3, "Default probes did not provide three consumed observations"
            await self.context.close()
        rows = self.probe["asynchronous_windows"]
        assert any(row["nonzero_advantage_tokens"] and row["metrics"]["policy/raw_grad_norm"] > 0 for row in rows)
        assert any(row["mode"] == "sc" and row["nonzero_advantage_tokens"] and row["metrics"]["policy/raw_grad_norm"] > 0 for row in rows)

    async def _train_loop(self):
        configure_coreweave_s3()
        spec = self.cfg.trainer.score_centering_performance
        assert spec.scope in ("controls", "coverage")
        self._performance_applied_updates = 0
        self._performance_fixed_advantages = False
        self._performance_record_evidence = False
        self._performance_driver_profile = False
        self._performance_driver_profiles = []
        self._performance_phase = "setup"
        self.probe = {
            "schema": 3, "scope": spec.scope, "workload": "snowball", "complete": False,
            "windows": [], "qualification": {},
            "source_selection": {"campaign_source_rows": OmegaConf.to_container(spec.source_rows),
                                 "source_data_sha256": spec.source_data_sha256},
            "runtime": {
                "lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[3] / "uv.lock").read_bytes()).hexdigest(),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "megatron-core", "transformers", "marin-iris")},
                "policy_parallelism": OmegaConf.to_container(self.cfg.trainer.policy.megatron_config),
                "sampling": OmegaConf.to_container(self.cfg.generator.sampling_params),
                "optimizer": OmegaConf.to_container(self.cfg.trainer.policy.optimizer_config),
                "timing": "host walls around existing completed RPCs; no extra CUDA synchronization; stage walls can overlap",
            },
        }
        if spec.scope == "controls":
            assert self.cfg.trainer.policy.optimizer_config.lr == 0
            assert self.cfg.trainer.policy.optimizer_config.get("min_lr", 0.0) == 0
        await self._init_weight_sync()
        await self._sync_policy_for_rollouts(reason="initial")
        self.probe["setup_timings"] = dict(self.all_startup_timings)
        self.prompts = []
        for index in range(128):
            messages, _, extra, _ = self.train_dataset[index // 4]
            prompt = extra.get("reference_prompt_ids")
            if prompt is None:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
            self.prompts.append(list(prompt))
        self.probe["prompt_sha256"] = digest(self.prompts)
        self.probe["prompt_lengths"] = list(map(len, self.prompts))
        self.save_probe()
        if spec.scope == "controls":
            await self.controlled_measurements()
        else:
            await self.coverage_measurements()
        self.finish_performance_report()
