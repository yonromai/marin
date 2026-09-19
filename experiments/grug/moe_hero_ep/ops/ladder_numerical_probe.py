# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prefill/decode probe for the trained Hero d1536 export."""

import argparse
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests
from iris.client.client import iris_ctx
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.iris import remote_inference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

EXPORT = (
    "s3://marin-us-east-02a/marin/exports/grug/rav-ladder-d1536/step-15128/"
    "hf-bf16-vllm/8cc7d8f1f49a387bc51058d0deb3cfe47cb843126465cbc1b22f3fe3f7f7261b"
)
CHECKPOINT = "s3://marin-us-east-02a/marin/grug/rav-ladder-d1536/2026.08.18/checkpoints/step-15128"
EXPORT_DIGEST = "8cc7d8f1f49a387bc51058d0deb3cfe47cb843126465cbc1b22f3fe3f7f7261b"
PROMPT_IDS = (128000, 791, 6864, 315, 9822, 374)
CONTINUATION_IDS = (12366, 13, 578, 3224, 374, 7559, 304, 11104)


def _model(gpus: int, gpu: str) -> ModelConfig:
    return ModelConfig(
        name=f"rav-ladder-d1536-diagnostic-{gpu.lower()}-ep{gpus}",
        location=EXPORT,
        tokenizer="marin-community/marin-tokenizer",
        apply_chat_template=False,
        resource_hint=ResourceHint(
            gpu={gpu: gpus},
            cpu=16 if gpus == 1 else 64,
            memory="96g" if gpus == 1 else "512g",
            disk="128g",
        ),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=gpus,
            max_model_len=4096,
            max_num_batched_tokens=4096,
            max_num_seqs=64,
            vllm_batch_invariant=True,
            vllm_use_flashinfer_sampler=False,
            vllm_extra_args=(
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--enforce-eager",
                "--no-enable-prefix-caching",
                "--gpu-memory-utilization",
                "0.9",
                "--max-logprobs",
                "64",
            ),
            auto_overrides=False,
        ),
    )


def _oracle() -> tuple[dict, dict]:
    manifest = json.loads((StoragePath(EXPORT) / "qualification-manifest.json").read_text())
    oracle = json.loads((StoragePath(EXPORT) / "jax-oracle.json").read_text())
    assert manifest["payload_tree_sha256"] == EXPORT_DIGEST
    assert manifest["raw_checkpoint"] == CHECKPOINT
    assert tuple(oracle["prompt_token_ids"]) == PROMPT_IDS
    assert tuple(oracle["generated_token_ids"]) == CONTINUATION_IDS
    assert len(oracle["selected_token_logprobs"]) == len(CONTINUATION_IDS)
    return manifest, oracle


def _completion(url: str, model_id: str, tokens: tuple[int, ...], count: int, rank: int) -> dict:
    headers = {"X-Request-Id": uuid.uuid4().hex}
    if rank >= 0:
        headers["X-data-parallel-rank"] = str(rank)
    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model_id,
            "prompt": tokens,
            "add_special_tokens": False,
            "temperature": 0.0,
            "max_tokens": count,
            "ignore_eos": True,
            "seed": 0,
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "logprobs": 64,
        },
        timeout=(30, 300),
    )
    try:
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return {
            "prompt_token_ids": [int(token) for token in choice["prompt_token_ids"]],
            "token_ids": [int(token) for token in choice["token_ids"]],
            "top_logprobs": [
                {int(token.removeprefix("token_id:")): float(value) for token, value in row.items()}
                for row in choice["logprobs"]["top_logprobs"]
            ],
        }
    except Exception as error:
        error.add_note(f"rank={rank}, status={response.status_code}, body={response.text[:1000]}")
        raise


def _wave(url: str, model_id: str, tokens: tuple[int, ...], count: int, gpus: int) -> list[dict]:
    ranks = range(gpus)
    with ThreadPoolExecutor(max_workers=gpus) as pool:
        futures = [pool.submit(_completion, url, model_id, tokens, count, rank) for rank in ranks]
        return [future.result() for future in futures]


def _observe(url: str, model_id: str, oracle: dict, gpus: int) -> dict:
    cached = _wave(url, model_id, PROMPT_IDS, len(CONTINUATION_IDS), gpus)
    prefill = [
        _wave(url, model_id, PROMPT_IDS + CONTINUATION_IDS[:step], 1, gpus) for step in range(len(CONTINUATION_IDS))
    ]
    expected = oracle["selected_token_logprobs"]
    ranks = []
    for rank in range(gpus):
        steps = []
        for step, target_id in enumerate(CONTINUATION_IDS):
            full_row = prefill[step][rank]["top_logprobs"][0]
            cached_row = cached[rank]["top_logprobs"][step]
            steps.append(
                {
                    "step": step,
                    "target_id": target_id,
                    "oracle_logprob": expected[step],
                    "prefill_top1": prefill[step][rank]["token_ids"][0],
                    "cached_top1": cached[rank]["token_ids"][step],
                    "prefill_logprob": full_row.get(target_id),
                    "cached_logprob": cached_row.get(target_id),
                    "prefill_top64_ids": list(full_row),
                    "cached_top64_ids": list(cached_row),
                }
            )
        ranks.append({"rank": rank, "steps": steps})
    return {"ranks": ranks}


def run(gpus: int, gpu: str, result_uri: str, validate_only: bool) -> None:
    if gpus not in (1, 8):
        raise ValueError("only EP1 and EP8 controls are supported")
    if gpu not in ("H100", "GB200"):
        raise ValueError("only H100 and GB200 controls are supported")
    configure_coreweave_s3()
    manifest, oracle = _oracle()
    model = _model(gpus, gpu)
    accelerator = AcceleratorChoice(
        platform=Platform.GPU,
        gpu_type=gpu,
        gpu_count=gpus,
        target_cluster="cw-us-east-02a" if gpu == "H100" else "cw-us-east-08a",
    )
    inference = inference_config_for_model(
        model,
        accelerator,
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    identity = {
        "checkpoint": CHECKPOINT,
        "export": EXPORT,
        "export_digest": manifest["payload_tree_sha256"],
        "vllm_source_commit": VLLM_GPU_RELEASE.source_commit,
        "gpu": f"{gpu}x{gpus}",
        "tp": 1,
        "dp": gpus,
        "ep": gpus,
        "serve_config": asdict(model.serve),
        "oracle_backend": oracle["evaluation_backend"],
        "oracle_decode_batch_size": oracle["decode_batch_size"],
        "oracle_decode_seq_len": oracle["decode_seq_len"],
    }
    if validate_only:
        print(json.dumps(identity, indent=2, sort_keys=True), flush=True)
        return

    destination = StoragePath(result_uri)
    if destination.exists():
        raise FileExistsError(result_uri)
    result = {"identity": identity, "job_id": str(iris_ctx().job_id)}
    failure = None
    try:
        with remote_inference(inference) as session:
            session.wait_until_ready()
            result["inference_job_id"] = str(session.jobs[0].job_id)
            result["observation"] = _observe(
                session.model.endpoint.url("completions"),
                session.model.endpoint.model,
                oracle,
                gpus,
            )
    except Exception as error:
        result["error"] = traceback.format_exc()
        failure = error
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"LADDER_NUMERICAL_PROBE={result_uri}", flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, choices=(1, 8), required=True)
    parser.add_argument("--gpu", choices=("H100", "GB200"), default="H100")
    parser.add_argument("--result-uri", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(args.gpus, args.gpu, args.result_uri, args.validate_only)
