# Compact attention and mixed backward products

The measurements below qualify the compact-attention implementation before the
additional MPS operand-routing change in `3b9b6233`. That later change broadens
lossless per-product staging to BFLOAT16 operands and eligible FLOAT16-left/FLOAT32-right
products, while retaining the compact kernel when staging would widen a dominant
operand. Its operator and policy regressions are `MetalCompactMpsRoutingTest` and
`MetalMixedMatmulPolicyTest`; release-head CI records the integrated checks. The older
full-size measurements below must not be attributed to that later routing change.

Status: local operator, full-size FLOAT32 numerical/checkpoint, full-context compact
oracle, compact real-data/typed FILE, CPU controls and matched timing checks are complete
for the workloads recorded below. CI must also pass on the proposed merge head; its
current outcome is recorded in GitHub checks. These results are not a whole-training
speed claim for FLOAT32 or a guarantee outside the tested configurations.

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

Positive means slower. These observations prompted the attribution controls below;
being close to zero or reversing order is not by itself proof of no code-induced regression.
Total timed real-data updates also differed: Prism 80.913 → 82.565 seconds, Nova
130.459 → 117.327 seconds and Loop 208.525 → 211.826 seconds. Those different-workload
totals are retained rather than cancelled against the warmed results. No FLOAT32 speedup
or universal timing-equivalence claim is made. [float32-results.json](float32-results.json) preserves
every process, individual step timing, native observation, GC and sampled-memory summary.

The original runner stopped before the first Loop timing JVM when its earlier training
confirmation expired. A separately authorized continuation completed only those six
unfinished processes, with identical binaries, helpers, workloads and resource guards.
The parent's stopped outcome remains preserved; it was not relabelled a complete suite.

A further permanent test covers compact FP64-reference comparisons at 1,024 tokens
with production 9/3 and 12/12 head configurations. It was initially compiled but unrun
at the handoff; the later physical execution is recorded below, separately from the
85 earlier Metal tests. The handoff build passed `mvn verify`: 2,341 total tests, 1,522 executed, 819 explicit
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

The FLOAT32 path preserves shader arithmetic and does not add a tensor cast or a second
saved-output allocation: `castToDevice` returns the same tensor for an identical dtype.
Host dispatch is not instruction-identical, however. The new `attentionBuffer` uses
`requireStorageBuffer` plus a separate `bufferDType` lookup where the previous
`requireBuffer` checked both in one lookup; dtype-aware pipeline selection also adds
host checks. Constant-time lookups are not zero work. Source inspection alone cannot
prove these checks have no cost or uniquely explain the measured sub-percent differences.

This supports a slot-position explanation but does not prove it. The falsifiable control is
a same-binary pair per family (old/old and new/new) under the identical helper and guards:
a second-slot penalty of roughly +0.1% to +0.4% with no version term supports the hypothesis;
a flat same-binary pair with a persisting positive new-minus-old term refutes it and would
point back at the candidate.

### Same-binary control result

Twelve processes ran on 31 October with the unchanged `QualificationTiming` helper, fresh
thermal waiter and original guards: per family, baseline/baseline then candidate/candidate.
Slot 2 minus slot 1 medians, identical binary each pair:

| Family | Baseline/baseline | Candidate/candidate |
|---|---:|---:|
| Prism | +3.068% | -0.322% |
| Nova | -1.425% | +1.518% |
| Loop | +0.625% | +0.012% |

The simple "second slot is slower" hypothesis is **not confirmed**: the sign is positive in
only 4 of 6 pairs. The control records observed same-code variation
under these conditions: identical binaries, back to back, differ by -1.4% to +3.1%, about
ten times the largest earlier new-minus-old observation (+0.352%). Same-slot new-versus-old
comparisons within this control were +2.549%/-0.824% (Prism), -1.716%/+1.218% (Nova) and
+0.168%/-0.443% (Loop), again straddling zero with magnitudes above the earlier gaps.
The earlier positive FLOAT32 observations are smaller than this observed same-code
variation, but that does not uniquely attribute each difference or prove that host
dispatch is cost-free. This is measured uncertainty, not an accepted percentage tolerance.
The IDE was open during this control; its conditions differ from the earlier IDE-closed
runs, so neither its absolute medians nor its variation define a bound for those runs.

## Full-size compact real-data checks

Six 100-update real-data processes (Prism/Nova/Loop × FLOAT16/BFLOAT16, 1,024 tokens,
batch 1, frozen seed-100 data, unchanged `NdTypedTraining` helper byte-identical to the
qualified classes) completed on the candidate with no guard event, no sampled swap change,
every loss finite, two-byte GPU weights with FP32 masters/moments confirmed at steps 1 and
100 (`storageExact=true`) and four positive FP32 gradient norms each. The V111 baseline logs
(runtime 77e1c2f8…, same data and helper) are the numerical reference. Compact fusion
intentionally removes intermediate 16-bit score/probability rounding, so bitwise equality
is not expected; the review threshold is 0.01 nat on held-out and mean training loss.

| Case | Mean loss Δ | Held-out Δ at 0/25/50/75/100 | Final held-out new/old |
|---|---:|---|---:|
| Prism FLOAT16 | +0.00008 | -0.00002 / +0.00005 / +0.00202 / +0.00024 / +0.00024 | 3.4719 / 3.4716 |
| Prism BFLOAT16 | +0.00363 | +0.00023 / **-0.03316** / -0.00524 / -0.00389 / -0.00799 | 3.4558 / 3.4637 |
| Nova FLOAT16 | +0.00000 | +0.00000 / +0.00005 / -0.00000 / -0.00006 / -0.00001 | 3.6611 / 3.6611 |
| Nova BFLOAT16 | -0.00001 | +0.00020 / -0.00011 / -0.00014 / -0.00021 / -0.00013 | 3.6607 / 3.6608 |
| Loop FLOAT16 | +0.00003 | -0.00002 / +0.00001 / -0.00002 / +0.00000 / -0.00005 | 3.6659 / 3.6660 |
| Loop BFLOAT16 | -0.00007 | -0.00014 / -0.00004 / -0.00017 / -0.00001 / -0.00042 | 3.6678 / 3.6682 |

Negative means the candidate is lower. Five cases stay within the threshold at every point.
Prism BFLOAT16 breaches it once, at step 25, with the candidate 0.033 nat *lower*. Its
per-step training losses agree with the baseline to within 0.002 nat through step 19 and
then diverge chaotically from step 20 (individual steps ±0.06 to ±0.10 nat in both
directions) while every later held-out point favours the candidate by 0.004 to 0.008 nat.
The divergence is consistent with the changed intermediate rounding and a different
optimization trajectory: BF16 has seven explicit fraction bits, and Prism has the widest
attention of these configurations. Independent full-context output/derivative oracles
passed and this run showed no nonfinite losses or worsening final held-out loss. A single
seed does not establish general quality improvement or rule out every possible quality
issue. The threshold breach remains an explicit investigated finding, not a pass of that bound.

## Full-size compact typed FILE checks

Eighteen cases (Prism/Nova/Loop × FLOAT16/BFLOAT16 × write / fresh self / actual FILE
resume) ran with the v38/v111 guard and helper classes byte-for-byte, the original 3 GiB
heap, 5 GiB native, 4.5 GiB RSS, any-swapout and 15 GiB plus per-mode disk reserve. All
18 completed with `allStateExact=true`; resume and self receipts match the write receipts
byte-for-byte and SHA-256. Saved file sizes are identical to the V111 baseline files
(180,265,869 / 199,437,994 / 200,767,792 model bytes and 1,261,818,635 / 1,395,987,650 /
1,405,285,753 training bytes); their SHA-256 values differ because the three pre-save updates
now run through compact fused attention. Peak RSS was 2.41 to 3.57 GB. Write payloads were
retired by the resume cases after verification, with receipts kept. Disk headroom for the
write reserve was obtained by clearing regenerable pip, Homebrew-download and application
updater caches only; no evidence root, frozen file or checkpoint was removed.

## Matched compact whole-model timing

Twenty-four 100-update full-size compact processes ran back to back in one session: for each
family and dtype, old-compact/new-compact then new-compact/old-compact, unchanged
`NdTypedTraining` helper, fresh thermal waiter before every JVM and the original guards.
Medians are over updates 11 to 100; native peak is the managed-buffer peak reported by the
runtime; FLOAT32 reference medians come from the same-session position control above.

| Family | dtype | Old → new ms (old-first) | Old → new ms (new-first) | Change | Native peak old → new | FLOAT32 same session |
|---|---|---:|---:|---:|---:|---:|
| Prism | FLOAT16 | 814.6 → 656.5 | 797.5 → 660.9 | -19.4% / -17.1% | 3.188 → 2.717 GB | ~395–407 ms |
| Prism | BFLOAT16 | 1149.6 → 923.7 | 1115.1 → 946.0 | -19.7% / -15.2% | 3.188 → 2.717 GB | ~395–407 ms |
| Nova | FLOAT16 | 1315.6 → 996.0 | 1281.9 → 989.1 | -24.3% / -22.9% | 4.043 → 3.196 GB | ~550–561 ms |
| Nova | BFLOAT16 | 1901.6 → 1410.4 | 1756.4 → 1398.1 | -25.8% / -20.4% | 4.043 → 3.196 GB | ~550–561 ms |
| Loop | FLOAT16 | 1900.7 → 1457.9 | 1920.7 → 1475.4 | -23.3% / -23.2% | 4.996 → 3.725 GB | ~785–790 ms |
| Loop | BFLOAT16 | 3204.7 → 2569.5 | 3139.1 → 2519.1 | -19.8% / -19.7% | 4.996 → 3.725 GB | ~785–790 ms |

The candidate is faster than the previous compact path in all twelve pairs in both orders,
by 15% to 26%, with native peak 15% to 25% lower; peak RSS stayed 1.64 to 1.94 GB. These
are same-session full-model measurements, not an operator extrapolation. Compact training
remains substantially slower than FLOAT32 on the tested Apple M4 configuration (FP16
about 1.6 to 1.9×, BF16 about 2.3 to 3.3× the FLOAT32 step). Remaining compact and mixed
products still include custom GPU routes rather than all using MPS; the aggregate timing
alone does not assign the entire gap to one route. No claim is made that compact dtypes
are universally a speed option. Their purpose includes
the two-byte weight storage confirmed above.

The IDE was open throughout; the comparison is within-session and within-pair only.

## CPU-only old-versus-new control

Twenty-four processes with Metal unused (the frozen `NdCpuTiming` helper selects the CPU
backend; helper classes compiled against both JARs are byte-identical), full Prism/Nova/
Loop parameter counts, sequence 64, seed 100, pinned data, FLOAT32, original CPU guards
(3 GiB heap, 7 GiB RSS, 64 MiB swap, any swapout, 10 min). Per family: an old/new
2 warm-up + 2 measured smoke pair whose four losses must be bitwise equal, then three
alternating AB/BA/AB 10 warm-up + 30 measured pairs.

- Smoke gates: all three families exact (4/4 losses each).
- Warmed pairs: all 40 losses bitwise identical between old and new in all nine pairs.
- Peak RSS: 2.95–3.23 GB old versus 3.11–3.23 GB new; no guard event.

| Family | Round 0 (old first) | Round 1 (new first) | Round 2 (old first) |
|---|---:|---:|---:|
| Prism | +2.226% | -7.470% | +0.362% |
| Nova | +0.502% | -5.343% | +0.156% |
| Loop | -0.068% | -1.438% | +0.153% |

Positive means the candidate was slower. CPU operator and optimizer implementations are
unchanged; the shared attention entry point retains its ordinary CPU fallback. All CPU
loss comparisons passed exactly. The second process was slower in eight of nine timing
pairs, regardless of which binary ran second; Loop round 0 is the exception. This suggests
an order-related effect but does not by itself identify a thermal cause or prove identical
speed. No consistent version-specific CPU slowdown was established by these controls.
RSS and each individual timing result remain disclosed rather than pooled away.

After all suites, local `mvn verify` with the Metal device present executed 2,347 of
2,371 tests (24 explicit opt-in skips), zero failures or errors, with every Java,
native-source and frontend quality and coverage gate passing (portable lines 95.04%,
branches 87.47%; autograd 99.82%/96.47%). The production JAR SHA-256 is unchanged at
`bafcde515ae9e439afdec72657ebb5a68d81dbdc9f294898615b99bcef139cd2`. Production code is unchanged
from the CI-verified commit `1018c22a`; later commits add the full-context compact oracle,
documentation and evidence. Raw per-process logs, guards,
manifests and analyses for every suite above are summarised in
[compact-results.json](compact-results.json).

Merge approval also requires hosted CI on the exact proposed head and explicit maintainer
authorization. No release, publication or real checkpoint changes were performed.

## Frozen identities

- Baseline JAR SHA-256: `77e1c2f8a02cf9277bbb98620f19b81f5a2462db49a261aaf995e4ed5b4ed9d7`.
- Timed candidate JAR: `2628281987b86b866da9984345d961db98f210decff24a077339f8470b284860`.
- Final selected-test JAR: `bafcde515ae9e439afdec72657ebb5a68d81dbdc9f294898615b99bcef139cd2`.

After timing, only native capability checks for SIMD width/thread limits changed in
production. All embedded shader-source hashes remain identical; the final library
was rebuilt and passed the 85 physical regressions separately. Raw per-process
timings, manifests, safety telemetry, parsers and failed attempts are preserved in
the qualification evidence rather than replacing an earlier failed run.
