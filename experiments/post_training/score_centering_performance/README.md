# Score-centering performance and periodic probes

The baseline and final candidate completed on East: 21 and 39 applied updates respectively. Qwen's final
startup check completed all 19 updates. Every owned job is terminal, automatic retries are zero, and provider
pods are absent. The final default is one probe per ten learner policy steps, with SC disabled.

This record qualifies the implementation proposed in [MarinSkyRL #968](https://github.com/marin-community/MarinSkyRL/pull/968).
SC stays disabled by default. Periodic probes collect bounded evidence for a hypothetical correction while ordinary
rollouts retain chosen-token-only TIS. The measurements use serialized native training cycles; they do not establish
production asynchronous throughput or a learning benefit. [Research #996](https://github.com/marin-community/MarinSkyRL/pull/996)
is retained separately.

## Source and controls

Both implementations include main `32edb00f784dac6995136d034ad7231a9b985bc0`, including DP chunk sharing from #999.
The product control is `325f819f94c3ed14a550ef5bc502fb6ff2d5b5d8`; the product candidate is
`f6b6ccb1d763486af190da0ff7005ed169285cdc`.
The frozen lock SHA-256 is `a9686c1263fae6e77f560c22ab8faa99b694747ac57e09f816ae7ce923c19f99`.
The pinned baseline and candidate diagnostic runtimes are
[`7aad2b37e`](https://github.com/marin-community/MarinSkyRL/commit/7aad2b37e134a26b1eb78a4375c1f400c25a3661) and
[`021666fc1`](https://github.com/marin-community/MarinSkyRL/commit/021666fc1b04d9e425321a6279ca19849b4da90f).
Their benchmark scaffolding is excluded from the product PR.

The timed native body and common memory/profile/reference helpers are shared. The
[current instrumentation comparison](evidence/instrumentation-native-chat-proof.json) checks their ASTs and lists
every diagnostic difference. Candidate chosen-token-only requests keep the production dictionary format, and
candidate positive-head raw requests use flat arrays; the [eight-case request proof](evidence/request-controls-proof.json)
checks those controls. The native task path uses chat serving and retains its existing transport.
The earlier Qwen startup run forced flat format even for width-zero requests; its timings do not support product
overhead claims. Its causal checks and exact same-engine tokens, text, head IDs and probabilities remain applicable.

The candidate also updates the actual engine's OpenAI sampling override outside timed cycles and records its
effective parameters after each cycle timer stops. Previously, changing the diagnostic trainer's temperature
left native chat at its initial 1.0. Every measured baseline window was declared and sampled at 1.0, so its
controls are unchanged. The candidate's compact 0.7 check now uses 0.7 on the engine as well as the learner.
The [actual serving-override proof](evidence/native-sampling-controls-proof.json) records the correction. The older
Qwen compact native 0.7 warmup is not used as matched-temperature evidence. A new Qwen-only chat A/B checks
tokens, message, source versions, chosen/head probabilities and actual decode suppression before timed windows.

Snowball uses its retained 67B/A2B SFT model, tokenizer revision `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`,
and seed-zero curriculum source. Exact identities and URIs are in the submitted YAMLs. The topology is four
8-H100 learner nodes plus one 8-H100 inference node, with learner PP2/EP8/DP16 and TP/CP/SP1.
The completed baseline and final candidate use `cw-us-east-02a`.
Inference uses DP8/EP8. Every mode has 32 prompt groups, four responses per group, and the same mixed response
lengths 64/256/1024/4096, padding, mask holes and one empty-loss row. Learning rate, optimizer, temperature,
publication frequency, source tags and update semantics are shared.

The canonical training Parquet has SHA-256
`df971071e94b3efaa8c39a97beb1020dca6240dff71651fe263f620fc57c5bcd` and 23,265 rows. The pinned AIME
grader rejects some retained references, such as `**C)** 32`. The first native baseline window kept retrying that
task. It was canceled after three calibration updates, with no completed native timing window. Its partial report
and allocation remain in the evidence. [Source preflight](preflight_source.py) selected the first 192 rows accepted
by their real configured graders, excluding ten incompatible references. Selection uses task validity and does
not use model outputs or reward values. The [selection](evidence/source-selection.json) includes every index and
exclusion. It contains 145 AIME, 28 Reasoning Gym and 19 GSM8K groups. Both implementations use the same subset;
workers verify the original file hash before selecting it. The [actual-loader receipt](evidence/selected-dataset-preflight.json)
checks 192 unique UIDs and exact equality between rendered and retained prompt token IDs. Source bytes are unchanged.

## Native measurement

| Mode | Capture and scoring | Optimizer |
| --- | --- | --- |
| `ordinary` | Chosen-token TIS; periodic probes disabled | Ordinary PPO/TIS |
| `capture` | Continuous K128 and old/current head scoring | Ordinary PPO/TIS |
| `sc` | Same continuous K128 path | PPO/TIS plus SC on every update |
| `armed` | Periodic probes available, with no observation due | Ordinary PPO/TIS |
| `probe` | One full prompt group captured; compact sampled positions rescored | Ordinary PPO/TIS; hypothetical correction summarized and discarded |

Each mode has one profiled warmup and five measured repetitions in rotating order. Measured windows exclude
profiling, memory resets and diagnostic report publication, but include the real task workers, S3 group store,
admission, DP object sharing, old-policy forward, optimizer work, normal metric emission and verified weight
publication. Allocation accounting includes everything excluded from the cycle timer. Timers can overlap.
Memory records include allocator peaks, device usage sampled every 0.1 seconds, and current/cumulative host RSS;
persistent caches and sampling can obscure shorter-lived or incremental peaks.

Three calibration updates precede timing. Independent full-vocabulary causal checks cover temperatures 1.0
and 0.7, real prefixes and a shifted-prefix control. Frozen-sampler replay and identical-token trainer rescores
retain engine mismatch and post-update drift separately. Native timing windows require one applied update,
no skipped update, positive finite gradient norm and stamped source age zero. The candidate's profiled probe
also checks compact causal positions at temperature 0.7. Each Snowball probe mode captures four 4096-token
responses, including the subsequently masked row, so collection cost is measured for a full long group.

Six further candidate updates use normal task rewards, masks, GRPO advantages and an interval of two to
demonstrate repeated periodic observations. The default-cadence estimate amortizes the measured full probe
cycle with measured armed cycles; it does not extrapolate scalar-summary time or sampled-position coverage.
One positive-head request can make vLLM compute top K for other rows sharing its decode batch.

[Analysis](analyze.py) reports response/loss throughput, cycle/update/sync times, memory and descriptive 95%
Student-t intervals on paired log times within an allocation. Comparisons between implementations use
independent-allocation Welch log-time intervals, with degrees of freedom rounded down for the critical-value
table. Five repetitions, changing weights, persistent caches and one allocation per implementation limit those
intervals. Hardware and placement variation between allocations is not estimated.

## Whole-cycle results and default cadence

The [complete paired summary](evidence/snowball-paired-summary.json) verifies the frozen lock, package versions,
parallelism, source subset, prompt hash, lengths, mask counts, effective temperatures and applied updates.
Each window contains 174080 response tokens and 164484 loss tokens in a 128×4549 sequence tensor.

| Mode | Baseline cycle s / response tokens/s | Candidate cycle s / response tokens/s |
| --- | ---: | ---: |
| Ordinary TIS, probes off | 58.553 / 2973 | 58.237 / 2989 |
| K128 capture/rescore, SC off | 111.818 / 1557 | 84.434 / 2062 |
| K128 capture/rescore, SC on | 109.780 / 1586 | 84.734 / 2054 |
| Probes armed, no observation due | — | 58.033 / 3000 |
| Complete probe cycle | — | 63.645 / 2735 |

Capture cycles fell 24.49% between allocations (descriptive Welch 95% interval 23.34–25.62% reduction), and
SC cycles fell 22.81% (21.06–24.52%). Ordinary cycles changed −0.54% (−1.56–0.49%). These are combined
implementation results, not isolated speedups for each CPU change. Main's transport changes are held constant.

Full SC adoption overhead within the candidate is **45.50%** (paired descriptive 95% interval 43.38–47.63%),
down from the baseline's 87.49% (84.41–90.58%). Capture alone costs 44.98% (43.28–46.70%). SC versus matched
capture is 0.36% (−0.29–1.01%); the interval does not establish zero correction cost. The previous readiness
result of 21.3% used a different raw/cohort loop before main's transport changes. It is historical context,
not a before/after control for this work.

A full probe adds 5.612 s over an armed cycle (95% mean interval 4.493–6.732 s). It captures four 4096-token
responses, then consumes 96 valid sampled positions after masking, 0.0584% of the cycle's loss tokens. Probe
and armed optimizer tensor byte counts equal ordinary TIS. The small scalar summary is not used as a substitute
for this complete-path cost. Armed versus off is −0.35% (−1.14–0.44%), consistent with timing noise.

The chosen default interval is **10**. One complete probe plus nine armed cycles gives a modeled average
58.594 s cycle: **0.61%** overhead versus off (−0.17–1.40%), or about 0.96% incremental cost over the armed
path alone. This does not prove an overhead below 1%. The modeled time between observations is 585.94 s
(9.77 minutes), with startup probing immediately. At equal cycle sizes the sampled coverage across ten cycles
is about 0.00584%; one prompt group is not an unbiased population sample. Admission rejection, trainable masks,
response lengths, mixed decode batches and asynchronous overlap change cost and actual observation intervals.
The diagnostic used forced due/non-due paths and interval two for repeated natural observations. Changing the
product's provisional default from 100 to 10 changes cadence only; those measured runtime paths are unchanged.

Candidate generation-wait means are 43.715 s ordinary, 67.382 s capture and 49.186 s probe. Input collation
is 0.096/1.077/0.123 s, old scoring 0.996/1.746/0.985 s, and policy training 3.658/4.532/3.564 s respectively.
Weight synchronization is about 6.6–6.8 s. Timer scopes overlap, so these fields are not an additive decomposition.
The main remaining cost is full head collection and native chat response construction/transport, followed by
continuous head scoring. Probes bound scoring/export while retaining the measured full collection cost.

| Candidate mode | Learner peak allocated / reserved / sampled device GiB | Driver current RSS GiB |
| --- | ---: | ---: |
| Ordinary | 54.558 / 59.160 / 76.301 | 7.236 |
| Capture | 54.611 / 58.365 / 75.459 | 7.954 |
| SC | 55.535 / 60.424 / 77.562 | 8.201 |
| Armed | 54.560 / 59.164 / 76.305 | 7.405 |
| Probe | 54.559 / 58.199 / 75.320 | 7.406 |

Inference peaks are 58.309 GiB allocated, 58.877 GiB reserved and up to 64.406 GiB sampled device usage.
The summary also retains per-role host RSS and cumulative host peaks. Allocator caches and 0.1-second sampling
limit incremental-memory comparisons; a smaller sampled peak does not establish a memory saving.

## What the repeated probes observed

Six further Snowball updates used normal task rewards, masks and finalized GRPO advantages. Three observations
each consumed 128 fresh-source positions with zero stale or unknown tags. Every update applied once and skipped
none. Observations were about 127–128 s apart at interval two. Current/old PPO log ratios remained zero at the
pre-update snapshot; the probes' fresh behavior/trainer absolute log-ratio means were 0.0483, 0.0417 and 0.0612.

| Observation | Actual mean absolute scalar correction | Unit-advantage absolute correction | Zero-advantage fraction |
| --- | ---: | ---: | ---: |
| native/0 | 0 | 0.001209 | 100% |
| native/2 | 0.000460 | 0.000798 | 0% |
| native/4 | 0 | 0.001678 | 100% |

Current retained mass ranged from 0.97484 to 0.99285, and behavior mass from 0.97549 to 0.99284. Per-observation
coverage was 0.0455–0.0531% of loss tokens. A separate NumPy FP64 replay agrees with the nonzero actual scalar
to 1.12e-7 and with the profiled 0.7 probe to 5e-9. Natural unit-advantage summaries differ by at most 1.98e-6;
FP32 cancellation near the 1e-6 tail floor explains the sensitivity. Tail-floor fractions were 4.7–26.6%.

The unit-advantage signal remains visible when actual advantages are zero. It does not turn that zero into a
training correction. Repeated correction or fresh mismatch can motivate a small matched learning comparison;
these few groups do not predict its outcome or measure model-parameter gradients. Stale samples mix engine
mismatch and weight drift, while unknown samples remain separate. No learning-benefit trial was run.

## Implemented CPU and transport changes

| Change | Evidence | Complexity and upkeep |
| --- | --- | --- |
| Skip head-text decoding in the actual native chat path | Baseline frontend profiles identify `_update_sample_logprobs` and tokenizer conversion as dominant Python work. [Pinned formatter/request-ID checks](evidence/native-chat-heads-cpu-proof.json) preserve exact IDs/scores and public responses | Small extension to the existing exact-prompt request hook; omit unused private head-byte lists; no serving transport redesign |
| Parse exact internal token IDs without per-head regex allocations | [4096-position K128 parser](evidence/native-chat-parser-cpu.json): median 0.43448 s to 0.27309 s, values identical | Small strict-ASCII prefix/suffix parser; duplicate, finite-score and tie guards retained |
| Process flat vLLM heads in arrays | [4096-position K128 postprocessing](evidence/cpu-flat.json): median 0.17634 s to 0.05314 s; values identical | Small adapter change; preserve duplicate, tie and requested-ID semantics |
| Skip decoded text for raw head entries | [Pinned vLLM processor on retained heads](evidence/cpu-head-decoding.json): 0.06907 s to 0.00336 s for 192 positions | Small existing request-hook extension, tied to the frozen vLLM build; response detokenization is unchanged |
| Preserve arrays and narrow safe IDs to int32 | Dense 128×4096×128 IDs occupy 256 MiB rather than 512 MiB; IDs plus two float32 head tensors occupy 768 MiB rather than 1 GiB | Small type/transport change; convert IDs to long at the GPU indexing boundary |
| Sort validation values without unused permutation indices | [Retained-head collation](evidence/cpu-final-collation.json): median 0.16551 s to 0.11515 s for eight 4096-token rows; outputs identical | One NumPy sort in the existing CPU-only collator |

These isolated CPU measurements identify work removed; their percentages are not GPU cycle improvements.
The vLLM integration uses the existing tagged-request hook against the frozen `39e62869693c` build; no wheel,
dependency or lock change is hidden in the launch. GPU and paired whole-cycle results determine practical benefit.

The native capture warmup profiles verify that head-token conversion calls fell from 174080 to zero. Across
eight frontends, `_update_sample_logprobs` cumulative time fell from 510.851 s to 33.252 s. These process timings
overlap and must not be added to cycle time. Native chat still builds and serializes millions of head objects.
Two [constructor alternatives](evidence/native-chat-constructor-cpu.json) had no CPU benefit, and a
[fixed-field subclass](evidence/native-chat-fixed-fields-cpu.json) changed a nested validation/serialization
microbenchmark only from 0.514 s to 0.499 s with overlapping repetitions. That small result does not justify
another private-validator hook. The existing validated constructor is retained.

Ranked by observed benefit and upkeep, the native decode/display-byte fix leads, followed by array preservation,
safe int32 IDs, strict ID parsing and values-only collation. Those are implemented within the existing interfaces.
A compact native head response or selective GPU-resident collection could address the remaining object/transport
cost but requires a larger engine/transport design and new parity qualification. That expansion was raised for
consultation and was not authorized; this record completes the tested small-patch scope. K32 remains an optional
approximation with no Snowball learning or parameter-gradient justification here.

## Failure and resource bounds

Every run has zero automatic failure/preemption retries and a unique handle. Qwen has a one-hour wall-clock
bound; Snowball has six hours from submission, with 600 seconds per native window in the repaired harness. One diagnosed manual
replacement per phase was declared. Qwen's first run exposed an initial source-version mismatch in the
benchmark; its replacement passed. The first East Snowball submission never received GPU admission and was
withdrawn before moving to RNO. RNO baseline a0 exposed the task-reference mismatch described above; a1 is
the diagnosed replacement. RNO a1 passed source and probability checks, then Kueue preempted it for a
three-node interactive Hero EP12 workload during the first native old-policy forward. All five tasks became
terminal and their pods disappeared, with zero automatic retries and no completed timing window. The next
finite capacity phase, East a2, uses the same runtime and validated subset. It has its own six-hour deadline
and zero automatic retries. It completed all 21 updates, all five tasks succeeded, and provider pods are absent.
Both clusters were full at submission; RNO also had a larger interactive burst
and more large checks announced. Active allocations are retained. Queued placement decisions check both clusters,
whole-node fit and admission state. Baseline and candidate timing comparisons stay on one cluster.

[Allocation accounting](accounting.py) uses controller timestamps and verified provider pod absence. The
provider upper bound includes submission/queue delay, startup, qualification, warmup, report/profile work,
idle time and teardown. The remaining controller time after native cycles is retained as a combined residual;
the available timers cannot split every category. Controller RUNNING alone does not prove GPU admission.
The withdrawn East submission has zero admitted GPU time. Original reports remain intact; compressed copies
and hashes are listed in [the report manifest](reports/manifest.json). The complete candidate uses a standard LZMA-compressed ZIP archive
 to satisfy the repository's existing 500 KiB file limit; decompression restores the original bytes.

The [final ledger](evidence/allocation-ledger.json) accounts for eight submitted jobs: **98.360 H100-hours**
from recorded admission or the earlier Qwen controller-start proxy, **99.114 controller-runtime H100-hours**,
and a conservative **110.275 H100-hour provider upper bound**. Failed starts and external preemption are included.
The successful Snowball baseline/candidate allocations used 29.350/36.758 H100-hours. Admission to initial sync
was 638.4/575.0 s, including 9.621/9.171 s in the sync calls. Native cycles including profiled warmup consumed
1776.4/2515.6 s; combined setup/qualification/report/idle/teardown residuals were 948.7/790.9 s. The residual
cannot be split further from retained timers. No dollar-cost claim is made.

The controller/provider [shutdown receipt](evidence/all-owned-shutdown.json) checks both H100 clusters, including
the earlier readiness jobs sharing this native ID. Only the allocations in this follow-up enter the ledger.
Other candidate YAMLs in this directory are prepared historical plans; only `owned_allocations` in
[the protocol](evidence/protocol.json) were submitted. Resource-board intentions are closed.

Reproduce analysis from this checkout with the product's frozen Python environment:

```sh
python experiments/post_training/score_centering_performance/analyze.py \
  --candidate experiments/post_training/score_centering_performance/reports/sc-performance-snowball-candidate-a3-01a146a3.json.zip \
  --baseline experiments/post_training/score_centering_performance/reports/sc-performance-snowball-baseline-a2-01a146a3.json.gz \
  --cadence 10 --output /tmp/score-centering-summary.json
python experiments/post_training/score_centering_performance/accounting.py \
  --protocol experiments/post_training/score_centering_performance/evidence/protocol.json \
  --report-dir experiments/post_training/score_centering_performance/reports \
  --output /tmp/score-centering-allocation-ledger.json
```
