# Campaign notepad

2026-10-10 03:00 UTC: Read the full overnight brief. Both allowed H100 clusters returned healthy current controllers. MarinSkyRL main is c96f6d25 and Marin main is c4687932. Earlier Snowball evidence at 40 updates and consumed ages 0–2 does not qualify the new port or settle severe staleness.

The released reference has no dependency lock. Freeze a resolved Python 3.13/CUDA12 JAX environment and record it separately from code. Reference evaluation holds out the first 64 rows after the training-seed shuffle, rather than using the environment's separate eval split. Its terminal step-299 evaluation is before update 300; this timing is now explicit in the protocol.

Initial state at 2026-10-10 03:00 UTC:

| Rung | Status | Evidence needed next |
| --- | --- | --- |
| Authors' reference | Preparing | Frozen environment; bounded real GPU update and retained outputs |
| SkyRL reproduction | Porting | FP32 parity, current-head GPU serving/update qualification |
| Qwen task/recipe transfer | Not started | Qualified reproduction followed by separately frozen changes |
| Snowball controlled mismatch | Not started | New-head five-node qualification and calibration |
| Snowball staleness | Not started | Measured 65-update cadence; frozen substantive main horizon |

The reference TIS and SC pilots each completed two full 64-by-8 updates on eight H100s. SC also exported its HF weights. Both serving captures were nonuniform; the natural top-128 head matched the sampled token probabilities at the exact next-token/prefix positions. These pilots qualify execution, not a learning effect.

Three earlier attempts remain failed engineering attempts: inherited Marin uv constraints broke strict hash installation; the image's old uv could not download a newly pinned Python patch; and the recorder needed explicit JAX output sharding to select two evidence rows. The fixes leave the released model, sampler and loss unchanged. The resolved reference environment is isolated from Marin config and uses the task image's Python 3.13.5.

Initial trajectories differed between runs despite equal seeds. Two additional successful diagnostic pilots established exact initial FP32 weight hash `ab4ac115c005a20165cc303e3b1b5f080e62f6b95d265a27e04f4af1e1439a43` and fixed-delta hash `74713004b950a61a105a6b8ba07b03e78113b08e8c3b0239f50d1e9b37ca1361` in both arms. Equal source, environment, initial weights and noise exclude those as the cause. Distributed BF16 execution and sampling outputs are not bit-identical in these pilots. Retain this observed variability; do not claim identical trajectories or infer a scientific result from two updates.

The current-main core at `117d0598` passed 257 objective checks plus 26 capture-control, replay and mesh checks. The released JAX loss and the production PPO/TIS closure agreed across twelve frozen FP32 cases, including actual +1/-1 group-centered advantages, masks, TIS cap 2 and modeled tails. Maximum gradient error was 2.24e-8. PPO clipping was inactive at p=o. Raw scalar losses have different origins and detached entropy terms; the artifact records both raw losses and their explicit scalar offset.

Countdown membership for seeds 0, 1 and 2 uses the exact original generator and shuffled held-out head. The first export's auxiliary prompt-ID field mistakenly stored BatchEncoding keys. No training used it. Preserve that export, regenerate integer prompt IDs as version two, and audit them against the authors' retained prompts before SkyRL launch. The original prompt text and membership hashes are unchanged.

The eight-run reference main matrix fixes 300 updates: three paired seeds at noise 0.05 plus one no-noise pair. It uses batch priority and does not occupy interactive campaign nodes. No main scientific result is available yet.

## Continuation at 2026-10-10T17:39:52.862506+00:00

| Rung | Current status | Evidence or next gate |
| --- | --- | --- |
| Authors’ reference | Complete | Eight full main runs re-audited; three noisy paired seeds and no-noise sanity pair |
| SkyRL reproduction | Partial | Older pilot audited; current-head paired qualification prepared, main matrix pending |
| Qwen task/recipe transfer | Not started | Reproduction must pass before changing the task and recipe |
| Snowball controlled mismatch | Not started | Five-node batch qualification and separate calibration pending |
| Snowball staleness | Not started | Actual 65-update serving cadence and substantive paired comparisons pending |

Noise0.05 paired mean clean-quality difference remains +46.61 percentage points, with n=3 paired-t 95% interval [-28.02,+121.25] points. SC seed1 remains stalled; the effect estimate is imprecise. Separate seed0 diagnostics give +0.20 points at noise0.01 and +33.59 points at noise0.02. They do not establish precise effects at those noise levels.

The older SkyRL pilot has two applied finite SGD updates with FP32 master weights and gradient buffers, zero momentum, zero decay and zero clipping. Complete stored behavior heads and tokens were audited, as described in the protocol. Exit0 alone was not used to qualify it. The detached native launch explains the missing terminal manifest; retained checkpoint2, resolved launch, raw groups, evaluations and Ray logs provide independent evidence. Needed temporary outputs are being promoted to durable storage.

Current-source CPU checks passed 78 tests after including the production loss module tests, which populate the policy-loss registry. Two isolated config checks had failed earlier because their registry was empty; no production or test tolerance was weakened. The bounded learner evidence and scalar persistence checks passed all 13 replay tests. New paired qualification configs pin published runtime `01944d73c92c493b835b9e3773147e4e72097e8f`.
