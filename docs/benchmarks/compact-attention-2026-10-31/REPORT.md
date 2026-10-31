# Compact attention and mixed backward products

Status: operator improvements validated; full-model qualification and exact-commit CI pending.
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

All completed operator and selected-model runs retained the bounded memory/disk/time
guards and recorded no sampled increase in global swap usage or swapouts. Short
processes without periodic RSS samples remain unsampled, not zero-memory processes.

Preserved failures: an initial reverse mixed MPS eligibility error aborted its test JVM;
the unsupported order was removed. Two autograd failures exposed broadcast gradients;
GPU packing corrected them. A larger integrated build encountered an obsolete test
expecting compact fusion to be unavailable and exceeded the incorrectly reused
single-operator process-tree RSS budget. That build remains failed. The capability
test now checks the supported dtype and still rejects strided storage. A fresh full
build retains the prior full-qualification safety envelope; it is not marked passed.

Still required: complete integrated native/desktop coverage, exact-commit CI,
full-size real-data training/stability, file restore, and matched whole-model
performance/memory acceptance. No release, publication or real checkpoint changes
were performed.

## Frozen identities

- Baseline JAR SHA-256: `77e1c2f8a02cf9277bbb98620f19b81f5a2462db49a261aaf995e4ed5b4ed9d7`.
- Timed candidate JAR: `2628281987b86b866da9984345d961db98f210decff24a077339f8470b284860`.
- Final selected-test JAR: `bafcde515ae9e439afdec72657ebb5a68d81dbdc9f294898615b99bcef139cd2`.

After timing, only native capability checks for SIMD width/thread limits changed in
production. All embedded shader-source hashes remain identical; the final library
was rebuilt and passed the 85 physical regressions separately. Raw per-process
timings, manifests, safety telemetry, parsers and failed attempts are preserved in
the qualification evidence rather than replacing an earlier failed run.
