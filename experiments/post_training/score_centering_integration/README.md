# Native score-centering integration

The native task adapter can carry K128 behavior heads as numeric arrays through the pinned vLLM engine API.
vLLM still renders messages, parses tools, samples tokens and handles aborts. Our adapter asks its public
`generate` interface for flat logprobs and attaches the selected arrays to the internal Ray response. It avoids
constructing, serializing and reparsing one chat object per candidate. Public HTTP chat responses keep their
existing shape. This record contains the completed Qwen interface qualification and the preregistered Snowball
comparison; Qwen and CPU measurements do not establish Snowball throughput.

The [prior performance evidence](../score_centering_performance/README.md) remains unchanged. The active
product is [draft #968](https://github.com/marin-community/MarinSkyRL/pull/968); #996 remains research.
The dependency stays at vLLM `39e62869693c46402b1a95fde4fc55ca1aab9ae1`. The qualified serving and token-version
hooks are unchanged. No new vLLM subclass, override, monkey patch, fork patch or dependency upgrade is used.

## Source and controls

| Purpose | Product | Instrumented runtime |
| --- | --- | --- |
| Qwen interface qualification | `d92307d3deff81aa8270c90c49d529a9bb7a1951` | `328858aa5ad1f659ee1bb6bf60aa6187be973512` |
| Snowball prior integration | `964a602e995b16943dc34db036d25d2b7c5b7310` | `bdb8b3134ca0b7322d889f2634dd1bfc6a126beb` |
| Snowball candidate integration | `db5227d4814086c5a71ca669b8c0dbb19563705b` | `a3cf30b6598de5a6de8ca45e720d5db8c47784d0` |

The later product commit `c44e390a5236079b116c28dc27f84db2ec90c2cb` adds a structural sampling type; its
generation and selection operations are unchanged. Runtime-only measurement code stays outside the product
PR. [The harness proof](evidence/matched-harness-proof.json) checks identical benchmark files, inference
measurement methods, lock and task image between the Snowball arms.

[The initial protocol](protocol.json) records the Qwen exposure. [The first Snowball protocol](snowball-protocol.json)
records final pins before its first submission; [revision a1](snowball-protocol-a1.json) extends future queue deadlines
after a provider preemption. Each Snowball arm uses five BATCH nodes, with four 8-H100
learners at PP2/EP8/DP16, TP/CP/SP1, and one 8-H100 inference node at DP8/EP8. It retains the Snowball 67B/A2B
SFT checkpoint and tokenizer revision from the prior evidence. The real grader preflight reproduced all 192
selected tasks, their order and their source hash, including the ten incompatible references excluded before
generation. No source rows were edited.

Each arm declares 87 applied updates: three calibration updates, five serialized modes with six rotating
repetitions each, six natural-reward periodic updates and four asynchronous modes with twelve updates each.
Serialized modes are ordinary TIS, capture/rescore with SC off, SC on, probes armed without an observation,
and a complete probe. One profiled warmup precedes five measured repetitions per mode. The controlled batch
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
attempt with zero applied updates. Its allocation is included in accounting. Baseline a1 is the second submitted
baseline attempt; the prepared candidate a0 was never submitted, and candidate a1 is its first planned attempt.

## Qwen interface qualification

The two-node RNO Qwen run completed 35 applied updates, all ten serialized windows, six natural periodic updates
and sixteen asynchronous updates. Exact old/new chat comparisons passed at temperatures 1.0 and 0.7 for both
128-token and 4096-token responses: token IDs, message, finish reason, source versions and chosen/head scores
matched. The compact response held int32 IDs and float32 scores, with display logprobs absent. Full-vocabulary
causal references, raw flat/dictionary parity and compact-position references also passed.

For a 4096-token response at temperature 1.0, a cross-node Ray echo carried 19270645 bytes as chat objects or
4268171 bytes as arrays. Three actor round trips were 0.678/0.683/0.676 seconds and 0.0211/0.0198/0.0200 seconds,
respectively. At temperature 0.7 they were 0.673/0.633/0.675 and 0.0195/0.0184/0.0189 seconds. These include actor
serialization, host copies and transport in both directions. They exclude generation, S3 and learner work;
halving or adding them to another timer would not measure a complete training cycle.

This Qwen run exercised the prototype's collection width of 128. An independent CPU BF16 boundary check later
found three of 32 cases where collecting one fewer natural candidate changed the identity of an equally probable
head token. The final implementation restores the old collection width of 129 and reproduces the chat display
limit before selecting K128; all 32 cases then matched exactly. The corrected check supersedes an invalid scratch
reference, as recorded in [the boundary evidence](evidence/chat-collection-width-tie-proof.json). Qwen's passing
four cases are retained as prototype evidence, not relabeled as a qualification of the final source pin.

The complete raw report is compressed without editing in [reports](reports/). Its original SHA-256 is
`6f8a49b8946a3ef7e481d51bc7bdbb916b89a6b11050046f8087dc86913b358c`.
[The manifest](manifest.json) retains source locations, sizes and hashes. Both Qwen tasks succeeded with attempt
zero; a fresh provider check verified their absence at 15:56:57 UTC on October 19.

## CPU evidence and remaining transfers

The final selector passed 441 CPU inference, trajectory and independent SC checks, with three dependency skips.
Another 25 checks cover probe scheduling, native rollout context and unchanged SC-off optimizer behavior.
The type-only follow-up passed the 124 targeted adapter tests. Tiny-training and CatCount explicitly disable
probes; their settings and sampling are unchanged.

A CPU representation experiment repeats 24 retained Snowball causal heads to 4096 positions at K128. It uses
the exact pinned Pydantic classes, formatter, qualified ID-only hook and prior parser. Across five alternating
measured repetitions after warmup, median representation/pickle/client time was 3.508 seconds for chat objects
and 0.0617 seconds for arrays; pickle sizes were 19210004 and 4231457 bytes. Inputs and returned numeric values
matched exactly. Flat primitive lists were prebuilt, so this excludes GPU probability collection, flat-list append,
real Ray/S3 and training. The shared host also limits interpretation of CPU timing.

vLLM's existing API still returns CPU `FlatLogprobs` lists. Our selector packs owned NumPy arrays, which cross
Ray to task workers. The existing rollout path projects them into aligned arrays, stores trainable groups using
its existing S3 pickle store, collates padded learner tensors, and transfers the required tensors to learner GPUs.
Those stages remain in complete native cycles. Response text still needs detokenization; chosen-only TIS uses
the existing chat path. No new serving or storage system is introduced.

The authors' [top-k collector](https://github.com/martin-marek/score-centering/blob/7c56e9ee2972aa57f446cf564de1a1658d14b321/models/topk.py)
buffers 32 decode steps on GPUs, batches local/global top-k selection, and reshares compact arrays into the
trainer layout. Their sampler and trainer share a GPU pool sequentially. Dedicated inference and learner nodes,
host lists, Ray and durable rollout storage impose different transfers here. The paper's custom-JAX under-1%
result does not measure vLLM integration overhead. Upstream
[AuxOutput logprob work](https://github.com/vllm-project/vllm/pull/60789) rejects online weight updates in its
documented contract and was closed on October 19; it is not adopted by this integration.

`analyze.py` checks complete worker reports and produces serialized, asynchronous, memory and cadence summaries.
It reuses the retained independent NumPy FP64 scalar replay and descriptive paired/Welch log-time comparisons.
Only exclusive StepWallTime fields partition a cycle. Nested timers and process CPU profiles can overlap.
Five repetitions and one allocation per implementation do not measure allocation-to-allocation variance.
