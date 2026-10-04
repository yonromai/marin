# Resolved score-centering run configurations

`current_qwen_qualification.yaml` is an October 4 source launch document for the
merged objective and rollout-buffer interfaces. It pins its launcher commit and
states the correction, normalization, KL and scheduling settings. Launch it from
the Marin checkout with the pinned `marinskyrl` package and
`configure_coreweave_s3()` before invoking `cloud.iris.launch`; the resolved document
is written to its durable `resolved_config_uri`. It is a runtime qualification,
not a confirmation quality experiment.

`current_qwen_qualification_initial_failed.yaml`,
`current_qwen_qualification_chatfix_failed.yaml`,
`current_qwen_qualification_capturefix_failed.yaml` and
`current_qwen_qualification_curriculumfix_failed.yaml` preserve the four failed
October qualification inputs. `current_qwen_qualification_hostmemfix_invalid.yaml`
preserves the successful exit whose post-probe behavior was uniform. The active
Qwen input pins the repair and completed five batches with nonzero gradients in
four of them. Failure records and costs are in the current study.

`current_snowball_qualification.yaml` uses the current port, the adopted Snowball
model and the format-corrected pool. It requests 32 learner H100s plus 8 inference
H100s, the full 4,096-token response cap and a fresh step-five checkpoint. A
separate continuation will verify updates six and seven after restore. Its
optional profiler records actual forward/backward CUDA kernels on both pipeline
stages. It pins the original named Marin template and requests 1,800 GB RAM per
node. `current_snowball_qualification_template_failed.yaml` and
`current_snowball_qualification_builtin_failed.yaml` preserve the two launches
that failed before training. The first lacked the named template; the second
failed the exact-chat capability check.

`current_qwen_token_versions.yaml` enables the new measurement on two separate
eight-H100 nodes, multiprocess vLLM, rolling batches and a lease-age limit of two.
Its five batches recorded exact token ages and mixed generating versions
on the real GPU runtime. Consumed token spans and the actual optimizer-update
ledger are archived under a fresh durable run prefix.

The other files retain the historical September inputs. Their removed API keys
are evidence of those runs and must not be used to launch the current runtime.

Each JSON file copies `resolved-skyrl.json` from the corresponding
`s3://marin-us-east-02a/marin/users/romain/checkpoints/async-rl/<run>/<version>/` artifact,
with one terminal newline added for repository lint. Its JSON values and Hydra arguments
are unchanged.
Each file contains the materialized Hydra arguments and the model/data references used
by its Iris child. `r23_r26.json` describes the shared PPO artifact continued by both jobs.
Run labels, artifact names, versions, and launcher source commits are in
[`score_centering_study.md`](../../score_centering_study.md).

After excluding only run-owned output paths and names, the cap-1.05 Qwen pairs
`r24/r25`, `r27/r28`, and `r30/r31` differ only in
`trainer.algorithm.score_centering_topk` (`0` versus `32`). Their TIS controls retain
`generator.sampling_params.logprobs=32`. Across pairs, the only training setting that
changes is `trainer.seed` (`17`, `18`, or `19`). The Snowball smoke pair likewise differs
only in `score_centering_topk` after excluding run-owned paths and names.

The active-cap near-fresh pair `r32/r33` differs only in `score_centering_topk` after
excluding its run-owned paths and names. Relative to the cap-2 near-fresh pair `r21/r22`,
its only training-setting change is `tis_imp_ratio_cap=1.05` instead of `2.0`.
The active-cap top-k-eight pair `r34/r35` likewise differs only in
`score_centering_topk` (`0` versus `8`). Relative to `r24/r25`, the TIS control changes
only `generator.sampling_params.logprobs` from 32 to 8; the SC arm also changes its
matching correction width from 32 to 8.
The delayed-publication pair `r36/r37` differs only in `score_centering_topk` (`0` versus
`8`) after excluding run-owned paths and names. Relative to `r34/r35`, the only training
settings changed are `max_staleness_steps` (`8` to `16`) and
`weight_sync_interval_steps` (`1` to `10`). Step-end evaluation still publishes the latest
weights at updates 10, 20, 30, and 40.

The cap-2 older pair `r19/r20` also differs only in `score_centering_topk`. Relative to
`r19`, near-fresh `r21` changes the age limit, generation-worker count, and finished-group
buffer; plain PPO `r23_r26` disables TIS and behavior-logprob capture; and `r29` changes
the captured behavior-logprob width from 32 to one. The pod host-memory request is an
Iris execution setting outside these Hydra arguments and is recorded in the study.

The six `confirm_seed{20,21,22}_{tis,sc32}.json` files are the resolved inputs for
the predeclared 40-update Qwen confirmation. Each within-seed pair has 153 Hydra
arguments. Excluding run-owned output paths and names, its sole difference is
`trainer.algorithm.score_centering_topk` (`0` or `32`). Within an arm, the only
material difference across seeds is `trainer.seed` (`20`, `21`, or `22`).

`qwen_fresh_version.json` is the resolved four-update age-zero diagnostic
input. It publishes weights after each update and admits only responses from
the current version, so versions one through three can use the consuming
trainer as their exactly matched B scorer before its next optimizer update.

`snowball_pilot_tis.json` is the resolved input for the aborted Snowball
full-response TIS attempt on the old `2026.08.29.1` pool. It has 154 Hydra
arguments, including the `dp_reshardable` optimizer checkpoint setting.
Its step-zero evaluation exposed a reward-format failure before the matched
quality comparison; see the study and format audit. The replacement pair uses
the format-corrected `2026.09.18` pool.

`snowball_formatfixed_tis.json` is the replacement TIS control's resolved
input. Its 154 Hydra arguments match `snowball_pilot_tis.json` after excluding
run-owned names and output paths. The launcher changed the pool reference from
`2026.08.29.1` to `2026.09.18`; that reference is recorded in the study and
Iris job, while the resolved SkyRL config contains only its staged local paths.
`snowball_formatfixed_sc32.json` is the replacement SC arm's resolved input.
After excluding run-owned names and output paths, its 154 arguments match the
replacement TIS control in order except for `trainer.algorithm.score_centering_topk`
(`32` versus `0`).

`snowball_full_pair_tis.yaml` and `snowball_full_pair_sc32.yaml` retain the
rendered inputs for the clean version `2026.09.21.7` pair. Their SHA-256 values
are `7430a3f72e3539c2e3ecd63258459ae60d6355b0b6b617c7f605ac62b8a50e42`
and `f0ba018e072542b8f72773e2aca4012a05ca9e81dc266e1595fe2346cf7e9156`,
matching the staged pod files. Their 118 flattened values differ only at
`trainer.algorithm.score_centering_topk` (`0` versus `32`).

The `snowball_full_pair_*_continuation5.yaml` files retain the rendered
1,536-token training-cap continuations. They keep the 4,096-token held-out
evaluation cap and resume the two arm-specific step-five checkpoints. Their
only other difference is `trainer.algorithm.score_centering_topk` (`0` versus
`32`).
