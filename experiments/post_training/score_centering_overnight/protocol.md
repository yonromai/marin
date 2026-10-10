# Score centering reproduction and Snowball transfer

Campaign owner: native devbox Codex session `01a123c0-7354-7993-8310-c03dd9b9c994`.
The [overnight brief](/home/romain/data/sessions/devbox/codex/01a0bb6f-0196-7e00-b2ad-90559d1641d9/prompts/score-centering-overnight-20261009.txt) defines completion. No positive result is required.

## Sources and gates

The reference is [the released code](https://github.com/martin-marek/score-centering/tree/7c56e9ee2972aa57f446cf564de1a1658d14b321), including its noise sweep and Countdown verifier. The core port starts from MarinSkyRL main `c96f6d25f59959096c3b179a51e67c0af6c99536`, carrying the simplified implementation at `4cb0d3d82a9833fb824083f018c61ee363795bfd`. Marin launcher main is `c4687932b36be236942ae2d4e259bb3864723e14`. Exact experiment heads and the resolved reference environment are frozen before matrix submission.

First run a bounded reference GPU pilot and a bounded SkyRL pilot through their real installation, launch, serving, capture, optimizer, and artifact paths. Frozen-input FP32 loss and gradient comparisons precede SkyRL training. Validate actual probabilities and token prefixes; reject uniform serving or missing evidence even if a process exits successfully. Preserve every failed attempt and its allocation time. Use at most four interactive H100 nodes across both clusters, including queued jobs; Snowball's five-node geometry always uses batch. Do not restart shared clusters.

## Rungs 1 and 2

Use Qwen/Qwen3-0.6B at a recorded HF revision, the exact seeded 20,000-example authors' Countdown generator, and the authors' rendered chat with `enable_thinking=false`. After shuffling by training seed, hold out the first 64 examples, as released `train_rl.py` does. Train on successive groups of 64 prompts with eight completions each. Maximum total length is 512 prompt-plus-response tokens. Sample at temperature one from the full distribution.

Use SGD at 0.01 without momentum, decay, warmup, or gradient clipping; FP32 master weights, BF16 forward; one full-batch optimizer update and token-mean loss. Advantages are solved/unsolved +1/-1 minus the group mean, without standard-deviation normalization. Capture natural top-128 probabilities in both arms. Compare token TIS cap 2 against identical TIS plus centering. Generate the noise once as 0.05 times an independent Gaussian times each INITIAL weight; every update copies fresh trainer weights and adds this same delta. Keep the trainer unperturbed.

Run 300 updates with collapse termination disabled and clean-policy evaluation every 20 steps. Preserve the released evaluator's timing: step 0 evaluates initial weights, and the final logged step 299 evaluates after 299 updates, before the final update. The primary reference endpoint is this released terminal evaluation's solved-response rate (512 responses over 64 held-out prompts). Record the completed-update count separately. Save the final 300-update weights and raw evaluations. SkyRL evaluations reproduce these weight states and membership. Describe JAX/PyTorch differences explicitly; gradient equivalence at p=o matters more than scalar surrogate offsets.

Use a paired runtime pilot before paired seeds 0, 1, and 2. The main noise runs and one matched no-noise sanity pair (seed 0) complete each reproduction. If the reference does not show the reported separation, retain it and exercise the published 0.01 and 0.02 alternatives before drawing a scientific conclusion. A null pilot does not terminate the ladder.

The replication unit is training seed. Report each paired difference, the mean, and a two-sided 95% Student-t interval over paired seed differences, with n=3 and its resulting imprecision explicit. Plot every seed and distinguish partial checkpoints from final endpoints. Do not treat responses or repeated evaluations as independent training replicates.

## Rung 3

Keep Qwen and the frozen controlled mismatch. First change only to the intended curriculum; then change optimizer, advantage scaling, batching, and response-length recipe in separately labeled comparisons. Use paired pilots to locate where a separation disappears; investigate loss/serving/data differences before expanding seeds. Freeze each comparison's membership, endpoint, and horizon before running its main seeds. Preserve the old curriculum artifact identities when applicable.

## Rungs 4 and 5

Use four 8-H100 learner nodes plus one 8-H100 inference node. Qualify the exact experimental head with real FlashAttention, pipeline/expert parallelism, nonuniform sampler probabilities, aligned evidence, finite gradients, and valid output batches. Use a separate short controlled-noise calibration to choose a useful stress level with usable rollouts. Freeze that level and the main horizon before at least three paired seeds.

For staleness, start calibration at the released setting 64, which refreshes every 65 optimizer updates. Measure actual generating versions and optimizer ages. Compare TIS cap 2 with TIS plus centering; include a matched frequently refreshed control and the ordinary incumbent to expose adoption cost. The main horizon must cover multiple stale-policy refresh cycles and calibrated mismatch accumulation. The published 1,200-update horizon is the starting reference, not a 40-update smoke. Predeclare the justified Snowball horizon after calibration and before main comparisons. Any change to cap 1.05 is a separately labeled recipe.

Use native clean-policy evaluation. Record quality, loss/gradient stability, probability-gap tails, capped-token fraction, omitted candidate mass, correction magnitude, consumed tokens, elapsed time, and attempt-aware allocated H100-hours. Report quality against updates, tokens, and compute. No acceptable quality-loss margin was chosen, so do not claim equivalence or acceptable loss.

## Publication and accounting

Keep the core disabled by default and separate the Countdown/noise/staleness experiments from it. Publish immutable configs, revisions, data/model identities, raw evaluations, capture evidence, analysis, plots, allocation ledger, and a concise notepad. Open an eligible core draft PR only after the exact head has the stated qualification. Preserve historical branches and PRs. Account for all campaign jobs, including failure and preemption, before declaring the full goal complete.
