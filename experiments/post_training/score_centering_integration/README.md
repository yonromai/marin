# Native score-centering integration

The native task adapter can carry K128 behavior heads as numeric arrays through the pinned vLLM engine API.
vLLM still renders messages, parses tools, samples tokens and handles aborts. Our adapter asks its public
`generate` interface for flat logprobs and attaches the selected arrays to the internal Ray response. It avoids
constructing, serializing and reparsing one chat object per candidate. Public HTTP chat responses keep their
existing shape. This record contains the completed Qwen interface qualifications and the preregistered Snowball
comparison; Qwen and CPU measurements do not establish Snowball throughput.

The [prior performance evidence](../score_centering_performance/README.md) remains unchanged. The active
product is [draft #968](https://github.com/marin-community/MarinSkyRL/pull/968); #996 remains research.
The dependency stays at vLLM `39e62869693c46402b1a95fde4fc55ca1aab9ae1`. The qualified serving and token-version
hooks are unchanged. No new vLLM subclass, override, monkey patch, fork patch or dependency upgrade is used.

## Source and controls

| Purpose | Product | Instrumented runtime |
| --- | --- | --- |
| Qwen prototype qualification | `d92307d3deff81aa8270c90c49d529a9bb7a1951` | `328858aa5ad1f659ee1bb6bf60aa6187be973512` |
| Qwen boundary-corrected qualification | `db5227d4814086c5a71ca669b8c0dbb19563705b` | `a3cf30b6598de5a6de8ca45e720d5db8c47784d0` |
| Snowball prior integration | `964a602e995b16943dc34db036d25d2b7c5b7310` | `bdb8b3134ca0b7322d889f2634dd1bfc6a126beb` |
| Current Qwen/Snowball candidate target | `87f13fcd82b4677545816e0789c685d10cf051a0` | `92d057c561b0893a3d922e42f221eb8d5b1e68fb` |
| Snowball staged async baseline | `964a602e995b16943dc34db036d25d2b7c5b7310` | `3b5de0c201a8ab7034aa53d9be3645d1e5372a8b` |

The later product commit `c44e390a5236079b116c28dc27f84db2ec90c2cb` adds a structural sampling type; its
generation and selection operations are unchanged. Runtime-only measurement code stays outside the product
PR. [The harness proof](evidence/matched-harness-proof.json) checks identical benchmark files, inference
measurement methods, lock and task image between the Snowball arms.

[The initial protocol](protocol.json) records the Qwen exposure. [The first Snowball protocol](snowball-protocol.json)
records final pins before its first submission; [revision a1](snowball-protocol-a1.json) extends future queue deadlines
after a provider preemption. [Revision a2](snowball-protocol-a2.json) moves the sequential comparison to East
after a second preemption and declares one additional baseline attempt. Each Snowball arm uses five BATCH nodes, with four 8-H100
learners at PP2/EP8/DP16, TP/CP/SP1, and one 8-H100 inference node at DP8/EP8. It retains the Snowball 67B/A2B
SFT checkpoint and tokenizer revision from the prior evidence. The real grader preflight reproduced all 192
selected tasks, their order and their source hash, including the ten incompatible references excluded before
generation. No source rows were edited.

The original whole-run protocol declared 87 applied updates: three calibration updates, five serialized modes with six rotating
repetitions each, six natural-reward periodic updates and four asynchronous modes with twelve updates each.
Serialized modes are ordinary TIS, capture/rescore with SC off, SC on, probes armed without an observation,
and a complete probe. Its interrupted East baseline retained one profiled warmup and two complete measured rotations per mode. The controlled batch
contains 128 responses of mixed lengths through 4096, 174080 response tokens and 164484 loss tokens, with
padding, mask holes and one empty-loss row.

Asynchronous modes use ordinary TIS, capture/rescore, SC and interval-ten probes. They retain real task rewards,
normal EOS, applied updates, per-update weight publication, full-batch admission, maximum source age four and
64 in-flight groups. Two warmup updates precede ten measured updates per mode. Continuous arm walls include
ordinary bookkeeping; profiles, forced memory synchronization, initial warmup and final context teardown stay
outside these windows. Prefetch at finite boundaries, ordered modes, changing weights and response variation
limit inference from a short comparison. No learning benefit is claimed.

The first Snowball attempt had a 10800-second queue-inclusive deadline. Future submissions use 21600 seconds,
with the same finite measurement windows, zero automatic failure or preemption retries and an initial manual
limit of two attempts per arm. Matched arms use the same cluster sequentially. Provider admission,
task state and verified Pod absence determine allocation accounting; Iris RUNNING alone does not prove admission.

Baseline a0 waited 40 minutes for RNO admission at 16:55:59 UTC. Kueue evicted its driver at 17:05:08 to admit
another workload; all five tasks became terminal by 17:05:41 and their Pods were absent at 17:11:26. No report
was written. The harness writes its first report before its first optimizer update, so this was a startup-only
attempt with zero applied updates. Baseline a1 was admitted at 17:49:11 UTC and evicted three seconds later;
its tasks were terminal by 17:49:51 and Pods absent at 17:51:24. It also produced no report or update.
After both old queues and Pods were verified stopped, baseline a2 was submitted to East at 17:55:35 and admitted
at 18:09:47. A priority-11 workload preempted it at 18:44:49; all tasks were terminal at 18:45:25 and Pods absent
at 18:56:34. It saved 20 applied updates and 17 complete serialized cycles. The next in-flight update is unestablished.
Config suffixes in the original protocols name revisions, not actual submitted attempts.
[Allocation accounting](allocation-accounting.json) includes failed starts, queue time and teardown bounds.

[The staged protocol](snowball-staged-protocol.json) was frozen before any candidate Snowball result. It retains
the baseline's last complete balanced rotation: 15 cycles, with warmup and two measured repetitions per mode,
through applied update 18. The two complete cycles from the interrupted next rotation remain in the original
raw report and are excluded from the primary matched comparison. The candidate controls stage repeats those
same 18 updates. Two measured repetitions give wider descriptive intervals than the original five.

Separate baseline and candidate asynchronous stages start from the same original SFT checkpoint. Each performs
three calibration updates, six natural periodic updates and 58 asynchronous updates, for 67 total. Ordinary,
capture and SC modes each have two warmup and ten measured updates. The interval-ten probe arm has two
warmup and twenty measured updates, spanning two cadence intervals. The original per-update
measurement functions, serialized loop and periodic body are AST-identical; see
[the staged harness proof](evidence/staged-harness-proof.json). No partial training checkpoint is recovered.
That initial plan kept all three future stages on East, sequentially, with at most two actual submissions each, a
21600-second queue-inclusive deadline and zero automatic retries. Product code, K128, SC application frequency
and the asynchronous admission/publication controls are unchanged.

Candidate controls a0 was admitted at 19:44:38 UTC, then preempted by a priority-12 workload at 19:49:18,
before its first report or optimizer update. All tasks were terminal at 19:49:52 and Pods absent at 20:02:01.
The remaining controls retry is prepared but held until East has room for five whole nodes and sufficient
admission quota. [The stage attempt record](snowball-stage-attempts.json) keeps this failed start and the
remaining limits explicit. Both asynchronous stages remain unsubmitted. No candidate Snowball performance
or correctness result is established yet.

[Revision r2](snowball-staged-protocol-r2.json) moves both primary comparison arms to RNO after a fresh check
found seven whole-node fits, 69 GPU/RDMA units of unused quota and no pending workloads. All old East jobs and
queues were already absent. It declares a fresh 18-update baseline controls stage and the remaining candidate
controls retry there, followed by the two 67-update asynchronous stages. The East prefix becomes secondary
evidence and is not pooled with RNO. The new short baseline stage allows two actual submissions; the candidate
controls limit remains two including its spent East attempt. [The additional Qwen protocol](qwen-floor-protocol.json)
declares one source-correction qualification, the third and final Qwen attempt, before candidate Snowball.

The corrected source reproduces the pinned chat formatter's -9999 score floor before token-ID tie ordering.
Without it, a head concentrated on one token could select different IDs when the remaining scores become ties
under the display floor. Two regression cases failed before the fix; 126 adapter checks now pass. Four independent
comparisons with the actual pinned Pydantic formatter, including sampled scores and the display limit, match exactly.
[The floor proof](evidence/chat-floor-after-proof.json) retains the source hashes and synthetic contract cases.
Raw-generation selection and the SC kernel are unchanged. Actual Qwen and Snowball qualification of this source
are pending; earlier qualified source pins remain separate.

## Qwen interface qualification

The boundary-corrected two-node RNO Qwen run completed 35 applied updates, all ten serialized windows, six natural periodic updates
and sixteen asynchronous updates. Exact old/new chat comparisons passed at temperatures 1.0 and 0.7 for both
128-token and 4096-token responses: token IDs, message, finish reason, source versions and chosen/head scores
matched. The compact response held int32 IDs and float32 scores, with display logprobs absent. Full-vocabulary
causal references, raw flat/dictionary parity and compact-position references also passed.

For a 4096-token response at temperature 1.0, a cross-node Ray echo carried 19270645 bytes as chat objects or
4268171 bytes as arrays. Three actor round trips were 0.740/0.737/0.789 seconds and 0.0218/0.0186/0.0185 seconds,
respectively. At temperature 0.7 they were 0.740/0.634/0.716 and 0.0201/0.0187/0.0180 seconds. These include actor
serialization, host copies and transport in both directions. They exclude generation, S3 and learner work;
halving or adding them to another timer would not measure a complete training cycle.

An earlier Qwen run exercised the prototype's collection width of 128. An independent CPU BF16 boundary check later
found three of 32 cases where collecting one fewer natural candidate changed the identity of an equally probable
head token. The final implementation restores the old collection width of 129 and reproduces the chat display
limit before selecting K128; all 32 cases then matched exactly. The corrected check supersedes an invalid scratch
reference, as recorded in [the boundary evidence](evidence/chat-collection-width-tie-proof.json). The final Qwen
run repeated all four comparisons at the corrected source pin and passed exactly.

Both complete raw reports are compressed without editing in [reports](reports/). The final report's original SHA-256 is
`4314e2b00324c86b4b689ae500b5d35e32b67247d39a882105f10543d79de49d`.
[The manifest](manifest.json) retains source locations, sizes and hashes. Both final Qwen tasks succeeded with
attempt zero; a fresh provider check verified their absence at 18:19:42 UTC on October 19.

An [offline precision audit](evidence/qwen-final-a1-probe-precision.json) retains a limit of the existing SC kernel.
An independent FP32 current-equals-old formula reproduces held probe scalars within 2.6e-9. FP64 evaluation of the
same quantized heads differs by as much as 0.002755 in an artificial temperature-0.7 warmup with 21 valid positions,
all at the modeled tail floor. Using the FP32 tail coefficient in otherwise-FP64 arithmetic reduces that difference
below 1.6e-8. Subtracting near-unit head mass makes this tiny tail sensitive to rounding. The kernel is byte-identical
to the baseline; native old/new heads matched exactly. No kernel, modeled tail floor, tolerance or K was changed. This scalar
precision observation does not establish a parameter-gradient error or learning impact.

## CPU evidence and remaining transfers

The final selector passed 441 CPU inference, trajectory and independent SC checks, with three dependency skips.
Another 25 checks cover probe scheduling, native rollout context and unchanged SC-off optimizer behavior.
The type-only follow-up passed the 124 targeted adapter tests. Tiny-training and CatCount explicitly disable
probes; their settings and sampling are unchanged.

A CPU representation experiment repeats 24 retained Snowball causal heads to 4096 positions at K128. It uses
the exact pinned Pydantic classes, formatter, qualified ID-only hook and prior parser. Across five alternating
measured repetitions after warmup at the corrected source, median representation/pickle/client time was 3.410 seconds for chat objects
and 0.0651 seconds for arrays; pickle sizes were 19210004 and 4231457 bytes. Inputs and returned numeric values
matched exactly. Flat primitive lists were prebuilt, so this excludes GPU probability collection, flat-list append,
real Ray/S3 and training. The shared host also limits interpretation of CPU timing.

vLLM's existing API still returns CPU `FlatLogprobs` lists. Our selector packs owned NumPy arrays, which cross
Ray to task workers. The existing rollout path projects them into aligned arrays, stores trainable groups using
its existing S3 pickle store, collates padded learner tensors, and transfers the required tensors to learner GPUs.
Those stages remain in complete native cycles. Response text still needs detokenization; chosen-only TIS uses
the existing chat path. No new serving or storage system is introduced.

The authors' [top-k collector](https://github.com/martin-marek/score-centering/blob/7c56e9ee2972aa57f446cf564de1a1658d14b321/models/topk.py)
buffers 32 decode steps on GPUs, batches local/global top-k selection, and reshards compact arrays into the
trainer layout. Their sampler and trainer share a GPU pool sequentially. Dedicated inference and learner nodes,
host lists, Ray and durable rollout storage impose different transfers here. The paper's custom-JAX under-1%
result does not measure vLLM integration overhead. Upstream
[AuxOutput logprob work](https://github.com/vllm-project/vllm/pull/60789) rejects online weight updates in its
documented contract and was closed on October 19; it is not adopted by this integration.

`analyze.py` checks complete worker reports and the declared complete rotations from the interrupted baseline.
It produces serialized, asynchronous, memory and cadence summaries.
It reuses the retained independent NumPy FP64 scalar replay and descriptive paired/Welch log-time comparisons.
Only exclusive StepWallTime fields partition a cycle. Nested timers and process CPU profiles can overlap.
Two measured Snowball repetitions per mode and one allocation per implementation do not measure
allocation-to-allocation variance. Control and asynchronous stages have separate allocations and training
trajectories; their timing samples are not pooled.
