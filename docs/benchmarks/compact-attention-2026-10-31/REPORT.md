# Compact attention and mixed backward products

Status: operator and full-size FLOAT32 numerical/checkpoint checks passed. Full-model
performance sign-off and compact real-data/checkpoint qualification remain open.
These results are not merge approval or a whole-training speed claim.

## Scope and implementation

Tests used an Apple M4 MacBook Air with 16 GB unified memory and JDK 20.
The baseline is the qualified merged ND runtime, not the older matrix-only implementation.

- Same-dtype FLOAT16/BFLOAT16 inputs can enter fused causal attention on supported devices:
  64-wide heads, complete 32-token tiles, automatic selection from 256 tokens.
- Compact inputs are widened in register fragments. Attention arithmetic, normalization,
  unrounded saved output and gradients remain FP32. The returned result is rounded once
  to the selected dtype. This is not native16 attention arithmetic or automatic autocast.
- Large narrow FP32-left/FP16-right products can use MPS. Selected large narrow
  FP32-left/BF16-right products use a scoped GPU conversion then FP32 MPS.
  The conversion is included in timings and managed-buffer accounting.
- Compact-left/FP32-right products retain the previous mixed GPU kernel. MPS does
  not accept that operand order directly; no improvement is claimed for dV below.

The compact fused algorithm avoids intermediate 16-bit score/probability rounding.
Its compact output/gradient hashes therefore differ from separate compact attention.
An independent scalar FP64 reference checks the unrounded algorithm and all input
derivatives under the existing FP32 attention bounds. No tolerance was loosened.

## Single-attention forward and backward

One 1,024-token causal attention operation, nine query heads, three shared KV heads,
64-wide heads, scale 0.125, identical representable inputs. Each independent JVM ran
10 warm-up and 30 measured repetitions; old/new order was reversed in the second pair.
Elapsed time includes host encoding and GPU completion wait, not only device execution.

| Input dtype | Old medians, ms | New medians, ms | Managed native peak, old → new |
|---|---:|---:|---:|
| FLOAT32 | 6.130 / 6.119 | 6.060 / 6.155 | 16,625,664 → 16,625,664 bytes |
| FLOAT16 | 19.280 / 18.962 | 5.299 / 5.307 | 125,865,984 → 15,839,232 bytes |
| BFLOAT16 | 21.473 / 21.327 | 5.275 / 5.230 | 125,865,984 → 15,839,232 bytes |

The compact peak reduction is about 87.4% for this operator, not total training memory.
FLOAT32 complete output/gradient hashes were identical across both implementations
and repeats. Its timing pairs straddle zero; that does not establish universal timing
equivalence or replace full-model no-regression checks.

## Mixed backward products

Three packed batches reproduce the grouped attention backward shapes. Products use
exactly representable bounded dyadic fixtures, independent scalar checks and complete
output hashes. Each entry shows forward-order / reverse-order process medians in ms,
after 10 warm-up samples and over 30 measured samples.

| Product | Mixed input | Old | New |
|---|---|---:|---:|
| dQ | FP32 × FP16 | 2.222 / 2.192 | 1.242 / 1.226 |
| dQ | FP32 × BF16 | 2.368 / 2.939 | 1.149 / 1.297 |
| dK | FP32 × FP16 | 2.337 / 2.320 | 1.285 / 0.902 |
| dK | FP32 × BF16 | 2.266 / 2.242 | 1.421 / 1.302 |
| dP | FP32 × FP16 | 3.018 / 3.019 | 1.678 / 1.694 |
| dP | FP32 × BF16 | 2.992 / 3.010 | 1.700 / 1.710 |

The previous dV route remains unchanged; its variable measurements are retained in
the raw evidence, not pooled into an improvement claim. Matrix-first-group variability
also remains recorded. Exact dyadic outputs alone do not prove unchanged rounding for
arbitrary real training gradients.

## Regression evidence and limits

The final runtime passed 85 selected actual Metal tests without failures, skips or
aborted cases. They include independent attention references, ND batches, grouped-head
gradients, tiny compact loads, full-range FP32 upstream gradients, broadcast gradients,
strided fallback rejection and original rounding regressions. All five model families
also repeated three 1,024-token training updates exactly for both compact dtypes,
including losses, gradients and updated weights. Those models use width 128 and
vocabulary 31; they are not the full-size 90M/100M models or a long real-data test.

The portable build passed 1,522 executed tests, zero failures/errors, with 817 explicit
hardware/desktop/opt-in skips. Java, native-source and frontend quality/coverage gates
passed. Skips are not reported as executed hardware checks.

The exact production commit `1018c22ad72dbfec4f2e7be1a60f392564dcdc7f` also passed
the clean CPU, Metal, native-instrumented, desktop, frontend and coverage build in
[CI run 41198131890](https://github.com/KirstenAli/DeepJ/actions/runs/41198131890).
Its optional performance/diagnostic jobs were skipped, not performance evidence.

## Full-size FLOAT32 checks

The final runtime completed 42 full-size processes on the same Apple M4 configuration:
18 numerical/checkpoint controls, six 200-update real-data runs and 18 timing runs.
Prism has 90,128,896 parameters, Nova 99,711,552 and Loop 100,375,680. All use batch 1,
1,024-token context and identical frozen inputs within each old/new pair.

- Every gradient, weight, first/second optimizer moment and all 16,777,216 logits
  matched exactly in the full-state comparisons, including actual file restore.
- Complete model/training checkpoint bytes and SHA-256 matched the qualified files.
- All 600 paired training losses, 15 held-out measurements, 12 positive gradient norms
  and 360 warm-up/measured timing losses matched exactly.
- Steady native storage and buffer counts matched. Maximum native training peaks were
  3,070,867,472 / 3,617,185,040 / 4,502,628,880 bytes for Prism / Nova / Loop in both
  versions. Two initial transient peaks were lower in the candidate; no overall training
  memory reduction is claimed for FLOAT32.
- No completed case triggered a resource guard or recorded a sampled global swap increase.

Each timing process used 10 warm-up and 30 measured updates. The second pair reversed
old/new order. These are individual paired median changes, not a pooled speed claim:

| Family | Pair 1 | Pair 2, reversed | Pair 3 |
|---|---:|---:|---:|
| Prism | +0.086% | -0.017% | -0.211% |
| Nova | +0.265% | -0.415% | +0.352% |
| Loop | +0.199% | -0.272% | -0.106% |

Positive means slower. These small positive observations remain subject to attribution;
being close to zero or reversing order is not by itself proof of no code-induced regression.
Total timed real-data updates also differed: Prism 80.913 → 82.565 seconds, Nova
130.459 → 117.327 seconds and Loop 208.525 → 211.826 seconds. Those different-workload
totals are retained rather than cancelled against the warmed results. No full-model
performance sign-off is claimed. [float32-results.json](float32-results.json) preserves
every process, individual step timing, native observation, GC and sampled-memory summary.

The original runner stopped before the first Loop timing JVM when its earlier training
confirmation expired. A separately authorized continuation completed only those six
unfinished processes, with identical binaries, helpers, workloads and resource guards.
The parent's stopped outcome remains preserved; it was not relabelled a complete suite.

A further permanent test now covers compact FP64-reference comparisons at 1,024 tokens
with production 9/3 and 12/12 head configurations. It has compiled but has not yet been
executed physically; it is not included in the 85 completed Metal tests above.
The handoff build passed `mvn verify`: 2,341 total tests, 1,522 executed, 819 explicit
hardware/desktop/opt-in skips, zero failures/errors, with all portable quality/coverage
gates passing. Its production JAR SHA remains identical to the full-size tested runtime.

All completed operator and selected-model runs retained the bounded memory/disk/time
guards and recorded no sampled increase in global swap usage or swapouts. Short
processes without periodic RSS samples remain unsampled, not zero-memory processes.

Preserved failures: an initial reverse mixed MPS eligibility error aborted its test JVM;
the unsupported order was removed. Two autograd failures exposed broadcast gradients;
GPU packing corrected them. A larger integrated build encountered an obsolete test
expecting compact fusion to be unavailable and exceeded the incorrectly reused
single-operator process-tree RSS budget. That build remains failed. The capability
test now checks the supported dtype and still rejects strided storage. That failed local
build remains failed; the later exact-production-commit integrated CI build passed separately.

## Full-context compact oracle executed

On 31 October the extended `MetalCompactFusedAttentionTest` ran physically on the M4:
8 cases, 0 failures/errors/skips, including 1024-token 9/3 grouped and 12/12 multi-head
FP16 and BF16 against the independent FP64 reference. Largest absolute deviations were
3.2e-8 (accumulated output), 3.0e-9 (query), 1.4e-8 (key) and 1.3e-6 (value) against
tolerances of 2e-5/3e-5. Rounded outputs matched the independently rounded accumulated
output exactly and every gradient stayed FLOAT32.

## FLOAT32 timing attribution

[timing-position-analysis.py](timing-position-analysis.py) re-reads the 18 timing
processes above and separates slot position from binary version. The second process of a
pair was slower in 7 of 9 pairs (mean +0.144%), while the candidate was slower in 4 of 9
(mean -0.014%). Every positive new-minus-old observation (Prism +0.086, Nova +0.265/+0.352,
Loop +0.199) came from a pair in which the candidate ran second; in the reversed round the
baseline ran second and was slower by +0.417%/+0.272% in Nova/Loop. Within-process drift
between the first and last ten measured updates ranged from -0.35% to +0.66%, larger than
any version difference. GC counts differed by at most one collection per process.

Source review of the FLOAT32 path found no added per-step work: `castToDevice` returns the
same tensor for an identical dtype, `ensureStorageBuffer` is `ensureGpuBuffer` minus one
check, and the native `bufferDType`/`requireStorageBuffer` calls are the same O(1) pool
lookups the previous `requireBuffer` performed.

This supports a slot-position explanation but does not prove it. The falsifiable control is
a same-binary pair per family (old/old and new/new) under the identical helper and guards:
a second-slot penalty of roughly +0.1% to +0.4% with no version term supports the hypothesis;
a flat same-binary pair with a persisting positive new-minus-old term refutes it and would
point back at the candidate.

Still required: the same-binary position control; full-size compact real-data/stability
and exact typed file restore; matched compact whole-model timings against old compact and
FLOAT32; CPU controls; and final report, portable and exact-head CI checks. No release,
publication or real checkpoint changes were performed.

## Frozen identities

- Baseline JAR SHA-256: `77e1c2f8a02cf9277bbb98620f19b81f5a2462db49a261aaf995e4ed5b4ed9d7`.
- Timed candidate JAR: `2628281987b86b866da9984345d961db98f210decff24a077339f8470b284860`.
- Final selected-test JAR: `bafcde515ae9e439afdec72657ebb5a68d81dbdc9f294898615b99bcef139cd2`.

After timing, only native capability checks for SIMD width/thread limits changed in
production. All embedded shader-source hashes remain identical; the final library
was rebuilt and passed the 85 physical regressions separately. Raw per-process
timings, manifests, safety telemetry, parsers and failed attempts are preserved in
the qualification evidence rather than replacing an earlier failed run.
