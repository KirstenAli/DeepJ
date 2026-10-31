# N-dimensional tensors: qualification summary

This report covers the source-branch ND migration, optional FLOAT16/BFLOAT16 storage,
native compact matrix paths and associated lifetime/performance fixes. It is not a
claim about the published `0.11.0-alpha` artifact or a universal speedup.

## Current review status on 31 October

PR #177 remains **draft and unmerged**. The residual host CPU finding below has
now been causally explained and removed; the sections dated 30 and 29 October are
retained as history and do not describe the current candidate.

Three stacked causes were found by tracing the cleanup, `zeroGrad` and gradient
accumulation lifecycle, each fixed with permanent tests:

1. `Parameter.zeroGrad` allocated a fresh host `float[]` per parameter per update.
2. After `releaseTemporaryResources()` dropped a gradient's device buffer, the
   host-less gradient was reused instead of rebound, so the next accumulate
   uploaded it through `ensureHost()` plus a clone and the gradient then carried a
   permanent host array that was filled and zero-scanned on every later update.
3. The device-zero replacement allocated and zero-filled a model-sized set of
   Metal buffers every update that the following accumulate immediately replaced,
   visible as a native peak exactly `parameters x 4` bytes above the original and
   as main-thread kernel time rather than Java user time.

Metal `zeros` are now lazy tensors with neither host nor device storage; uploading
a host-less tensor queues a device zero-fill; and the zero-input skip recognises
host-less storage. A zeroed gradient therefore costs nothing until its first
accumulate writes one output buffer.

Matched original-pre-ND versus candidate Nova FLOAT32 synthetic 200-update runs
(old/new/new/old independent JVMs, fixed compiler workers, minimal observer)
retain exact bits for all 800 losses. Ordered candidate-minus-original results:
host process CPU **-11.77 s / -11.04 s** per 200 updates, main-thread user CPU
-6.54 s / -6.15 s, main-thread kernel CPU -0.29 s / +0.13 s, retired instructions
-73.9 G / -73.5 G (about -32%), all-200 elapsed -6.66 s / -8.13 s, steady
allocation 3.09 versus 3.24 MB per update, native peak identical. No swap increase
or swapout occurred. This is one controlled synthetic comparison; it closes the
specific host CPU finding and is not a universal speedup claim.

The three-family FLOAT32 real-data endurance check (Prism, Nova and Loop; 200
updates, five validations, four gradient norms each) was rerun for the current
candidate: all 209 recorded loss, validation and gradient lines per family match
the pre-ND reference exactly and native peaks are byte-identical.

The exact-commit hosted CI run and the local integrated full-coverage build both
passed on the pushed head: 2,345 tests, zero failures/errors, 12 explicit skips
(11 disabled manual timing tests and one optional user-checkpoint smoke test).

Outstanding before approval: rerun the typed FLOAT16/BFLOAT16 real-data, complete
FILE write/resume and full-state suites against this exact head rather than the
older frozen candidate, and complete the human performance-acceptance sign-off.
No merge or release is approved.

## Historical review status on 30 October

PR #177 remains **draft and unmerged**. The additional host CPU requirement was
reopened; the earlier acceptance below is historical and does not approve the
latest changes. The tested fixes are being made available for review, not release.

The latest candidate removes repeated immutable shape/layout work, preserves the
original FLOAT32 zero-upload entry point, avoids repeated random-fill accessors,
and replaces separate native buffer/dtype maps with one coherent buffer record.
Permanent tests retain validation order, overflow/zero behavior, alias mutation
checks, dense-layout bounds, all storage dtypes and native buffer-ID reuse.

The 30 October clean integrated build passed **2,320 of 2,332 tests**, with zero
failures/errors. The 12 explicit skips are 11 disabled manual timing tests and one
optional user-checkpoint smoke test; actual Metal, instrumented native, desktop,
frontend, CLI, quality and all coverage gates passed. All 82 Metal kernel
selections were covered. The production JAR's extracted entries match the frozen
candidate exactly; these report/receipt updates were made afterward.

Matched original-pre-ND versus current Nova FLOAT32 synthetic 200-update runs
retain exact bits for every loss and identical tracked native live/peak/counts.
Under the normal JVM policy, ordered host CPU changes are **+0.471%/+0.733%**;
the thirty measured updates after ten warm-up updates have elapsed changes
**+0.059%/+0.208%**. These are different metrics. Neither a small percentage nor a
correctness pass closes the resource/performance requirement.

A complete fixed-compiler-worker diagnostic found positive late noncompiler CPU
in both orders. A later clean-background repeat separated stored main-thread
user/kernel counters but did not establish one causal explanation: whole-run
noncompiler CPU changed -66.299 ms/+100.894 ms; late noncompiler +85.747 ms/-35.455
ms; late main user +15.886 ms/-9.308 ms and kernel +17.645 ms/+5.041 ms. All 800
losses, worker identities and native totals matched, with no sampled swap
increase or swapout. This diagnostic does not replace normal-policy acceptance.

One preceding attempt failed the unchanged global-swap guard after background
conditions changed. It remains failed; system-wide swap telemetry does not prove
which process swapped. No tolerance, safety guard or production compiler policy
was relaxed to obtain a pass.

Outstanding: causally resolve the host CPU finding, establish the full final
all-model/state/file/dtype/real-data/endurance/memory/performance matrix for the
final integrated candidate, and verify exact-head CI. Older candidate results
remain useful evidence but are not silently transferred to changed code. See the
[current review receipt](v104-review.json). No merge or release is approved.

## Historical local status on 29 October

At the V38 stage, local qualification was accepted for the tested training workflows
under the then-agreed criterion: correctness, safe memory and no demonstrated training-speed
regression, with additional Java compilation CPU disclosed separately. This does not
claim identical compilation cost, energy use, startup latency or every-step timing.
That acceptance was subsequently reopened as stated above. The following results
describe that older frozen candidate, not final approval of the latest fixes.
The frozen V38 candidate passed the complete local build: 2,297 tests, zero failures
or errors, and 12 intentional skips. Actual Metal, native instrumentation,
desktop/frontend, quality and coverage checks are included.

Final-source qualification now also includes:

- Original 200-update FLOAT32 workloads: 600 training losses, 15 held-out results
  and 12 positive gradient norms match the original baseline exactly.
- All 18 full-size FLOAT32 state/file cases: every weight, gradient, optimizer
  moment and full logit value, plus complete saved/resaved bytes, matches exactly.
- Six FLOAT16/BFLOAT16 100-update workloads: all 600 losses, 30 held-out results
  and 24 gradient norms match the original typed references exactly.
- All 18 final FLOAT16/BFLOAT16 write/self/actual-file-restore cases: complete
  state and logit digests and saved/resaved files match the original typed references.

These cases recorded no sampled global swap increase. Typed weights remain two
bytes per value and gradients, master weights and moments remain FLOAT32. Verified
disposable test files were retired only after byte/SHA and restore checks; receipts,
inputs, logs and failures are retained. Real training checkpoints were not changed.

The disclosed performance trade-off is **extra CPU work during GPU training**, not
a measured 4.9% training-time slowdown. In the latest original-baseline Nova1024
normal-JVM pairs, elapsed training time changed by -1.191%/-0.009%, while total CPU
time changed by -0.029%/+4.887%. Compiler CPU was higher in both pairs. The positive
CPU result is not averaged away, described as coincidental or claimed to be eliminated.
Its acceptance as a disclosed resource trade-off is distinct from the training-speed
checks; checkpoint and numerical passes alone would not establish performance.

The latest Nova comparison retains both orders and every phase:

| New relative to original pre-ND | Old-first pair | New-first pair |
|---|---:|---:|
| First 10 training updates | -3.654% | +0.163% |
| Next 30 training updates | -1.191% | -0.009% |
| All 40 training updates | -1.842% | +0.035% |
| Complete process, including diagnostics | -2.662% | +0.165% |

These short-window observations do not prove universal equality or a general speedup.
The positive observations remain visible alongside original uninstrumented timings,
same-code controls and causal CPU/GPU work investigations below. No numerical bound,
original workload or production compiler policy was changed for acceptance.
Two further private allocation optimizations were rejected after real-model CPU cost
increased; neither is included in the qualified source.

The [final qualification receipt](v38-closeout.json) records runtime hashes, complete
state/file gates and both timing pairs without local machine paths or private data.

## Reopened review

PR #177 is draft. The V33 results below remain historical evidence, not sign-off
for the latest local edits. Further investigation found and fixed padded batch
addressing: on the tested Metal backend, a batched matrix call could read the next
sliced matrix at `rows * rowStride` instead of the larger batch stride. Padded
batches now use explicit per-matrix offsets; ordinary packed batches retain their
batched path. Nine additional dtype/layout/gradient cases cover this defect.

A separate dense-layout address shortcut preserves all recorded compact training
and state comparisons. Two ordered warmed Nova1024 pairs per compact dtype show
14.1–14.3% less step time than the preceding ND implementation, not than pre-ND
FLOAT32. The private Java/compiler experiments from that stage were not adopted.
The V35 FP16 route passed a local 2,291-test build; the current V38 build above
also includes the two large-matrix precision regressions and subsequent optimizer
and buffer-binding tests. Exact-head CI remains required for the latest edits.

The local candidate now selects MPS for eligible larger FLOAT16 and mixed
FLOAT32-by-FLOAT16 products, without narrowing inputs or allocating conversion
shadows. Small-shape precision probes rejected an initial unrestricted route:
shape-dependent summation changed some output bits. Independent double references
isolated rounding rather than addressing errors. The selected route retains the
original small-product path; permanent tests reproduce two failures without this
restriction and pass with it. Mixed products additionally retain the original path
below the measured size threshold because MPS host setup can dominate small work.

The final restricted route repeats all three affected full-model runs: all 300
losses, 15 validations, 12 gradient norms and storage checks match the original
references exactly. Nine actual write/self/restore cases also match every state
digest, all 16,777,216 logits per case and complete saved/resaved file bytes.
Native retention remains stable, with no sampled global swap increase or guard event.

Two warmed Nova FLOAT16 pairs measure **25.5–25.8% less training-step time** than
the preceding ND compact path: 52.222 versus 38.762 seconds, then reverse-order
52.053 versus 38.772 seconds, for thirty measured updates after ten warm-up updates.
Every loss bit, data cursor and live-buffer count matches. This supersedes the
unrestricted prototype's approximately 32% result; it is not a published-release,
pre-ND or FLOAT32 speed comparison. Old-old total variation was -0.324%; candidate
self-variation was +0.027%. Neither pair is averaged against the other to hide a
slowdown. Complete per-step and memory observations are retained.
The FLOAT32 wall-time control was mixed; native instrumentation confirms its
11,297 commands never enter the changed matrix path, with identical counts and
loss bits. This does not establish universal wall-time equality.

## Reproducibility and scope

Tests ran on an Apple M4 MacBook Air with 16 GB unified memory and Temurin 20.0.2.
Full-size models were Prism 90,128,896, Nova 99,711,552 and Loop 100,375,680 parameters.
Metal comparisons used 1,024-token contexts; explicitly bounded CPU comparisons
used 64 tokens. All families also have smaller independent operator/model tests.
Seed, data, optimizer, precision and cleanup policy were matched between versions.

Original pre-ND JAR SHA256:
`bfbadaf4c1fc3d3241fef5b81fdf611767d5f61926204b4e1021ea97c5a3d591`.
Main migration/performance qualification JAR SHA256 (V29):
`810da7396c277cc6be09c3aa84a5b60162f62fa3a5291c316429facfd9362ef8`.
Corresponding frozen source archive SHA256:
`06895eb7b1da83525e79f424b58dd62f3b3073c54195fb3499b923064e1b986e`.
The historical V31 source audit compared all 781 source/test/native/script/configuration
files with that archive; all bytes matched. Afterward, documentation was updated
and one trailing blank line was removed from a test file. The subsequent hosted-device
fix below changed only the native library: all 459 other runtime entries, including
every Java class, remain byte-identical. Historical V33 JAR SHA256:
`83f6779f9ac6cebbd649ce570155326390f0e2bee8b9a1251155a24fb1dae958`.

Historical V35 JAR SHA256:
`ae24add81e44377c39408867b98c98e2768b6ffd43b86e2b707dfc0e9d22d472`.
All 456 other runtime entries, including every production Java class, match V34;
only the native library changed. The final warmed comparison's V34 baseline is
`5b706e54588c9c539454873a57436d1154f0796e1f488f9d0a707c6f321a2f73`.

Current frozen V38 JAR SHA256:
`a80914e4cf97ccd53501e96874e1acf609081f0757f5860bc906471f013b5093`.
V37 changed optimizer moment lookup/cold creation without changing arithmetic;
V38 reuses an existing weak buffer binding only when it refers to the same live
physical storage. All 456 other runtime entries match V37. The native library is
unchanged from the qualified restricted V35 route.

Raw inputs, frozen binaries, failed attempts, complete logs and telemetry are kept
with the local qualification evidence. This public summary contains no training
corpus, user checkpoint, generated private text or machine-specific filesystem paths.

## Correctness and state

- Independent primitive references and finite differences cover shape/layout,
  broadcasting, aliases, repeated indices, dtype conversions and gradients.
- CPU and actual Metal tests cover all five decoder families, independent sequences
  versus dense batches, shared Loop parameters, masks and generation sessions.
- Full-size FLOAT32 controls compare every gradient, weight, optimizer moment and
  full 1,024-by-16,384 logit value, plus actual file restore and continuation.
  The recorded baseline/self/fresh/file comparisons match exactly.
- The final full-size FLOAT32 endurance runs retain 600 paired training losses,
  15 held-out validations and 12 positive gradient checks, matching the reference.
- Six full-size compact-dtype runs retain 600 training losses, 30 held-out checks
  and 24 gradient norms, with stable native retention. These are separate 100-update
  dtype runs, not replacements for the original 200-update FLOAT32 workloads.
- All 18 full-size compact write/self/restore cases compare complete state and
  full-logit digests, then complete resaved model/training bytes. They match exactly.
  Actual weights occupy two bytes per element; masters, moments and gradients are FP32.

The historical V33 local full-coverage build ran 2,262 tests: 2,250 passed, zero failures or
errors, and 12 explicit skips. Eleven are disabled manual timing tests; one is an
optional local-checkpoint smoke test requiring `deepj.modelPath`. Actual Metal,
native instrumentation, desktop/frontend, quality and coverage gates passed.

## Hosted-device closeout

Hosted CI exposed two additional issues: compact matrix support was rejected before
probing compilation on virtual Metal devices, and fast compact normalization could
cross a FLOAT16 rounding midpoint. Capability probing and exact midpoint regression
tests now cover these cases. Precise compact normalization is selected for compatibility
devices; qualified Apple-family arithmetic is preserved. FLOAT32 expressions and
default optimized kernels are unchanged. This is a device-capability selection, not
CI detection or a tolerance adjustment.

An initial broader precision change passed unit tests but changed M4 training values.
The unchanged full-model comparison rejected it. That attempt remains a failed
qualification; it was narrowed and retested, not accepted with looser comparisons.

The final V33 runtime repeats all six full-size compact 100-update runs: every one of
the 600 recorded losses, 30 held-out results and 24 gradient checks matches the original
reference exactly. Native retention is stable, storage remains two bytes per weight,
and there is no sampled global swap increase or safety-guard event. Six additional
full-size fresh-state controls match all weight/gradient/moment/master digests and
all 16,777,216 logits per case against the earlier verified FILE receipts. These are
fresh-state comparisons, not claims that the earlier 18 actual FILE runs were repeated.
Persistence classes are byte-identical, and current local/hosted tests also exercise
typed save/restore. See the [machine-readable closeout](v33-closeout.json).

[Historical V33 exact-runtime-head CI](https://github.com/KirstenAli/DeepJ/actions/runs/40747428712)
passes CPU, fallback, actual Metal, desktop/frontend, native instrumentation and
all quality/coverage gates, including 100% Metal kernel selections. Earlier failed
CI, disk-guard stops and rejected numerical attempts are preserved in local evidence.

## Defects found and corrected

Qualification exposed real defects rather than just measurement variation:
scoped-view ownership, unnecessary device copying, dense ADD dispatch, checkpoint
initialization/staging and evaluation-output lifetimes. Regression tests now cover
these paths. CPU migration also introduced serial dense pointwise/SwiGLU work and
checked geometry/storage access inside inner loops. Parallel dispatch was restored,
invariant checks were hoisted, and avoidable shape/broadcast metadata was removed.
No numerical tolerance, production forced-GC policy or JVM compiler workaround was
added to make a result pass.

Early native peak differences were traced separately. Final V38 logical buffer
requests and retained memory matched, but two early cumulative peaks differed.
Twelve short old/new controls isolated both GC-dependent weak scratch reuse and
orphan retirement. Controlling both produced identical allocation plans, release
ledgers and peaks; the original code reached the flagged peak envelope too. This
closes that targeted memory finding without changing production GC or retirement
policy. Original failed peak comparisons and historical swap failures remain
recorded failures. System-wide swap observations are not attributed to one process
without process-specific evidence.

## Performance investigation

Original uninstrumented old/new timings and reversed/self controls are preserved.
Small positive timings were not waived using an arbitrary percentage margin.
Supplemental diagnostics isolated operation costs, allocation, native commands,
actual GPU time, main/process CPU work, compilation and process-order variation.
The historical diagnostics below investigate specific gaps; they do not replace
the original workloads, eliminate the disclosed compilation cost above, or justify a
universal speed guarantee.

Two independent full-parameter Loop CPU64 pairs alternated updates seconds apart,
with only one JVM training at a time. The peer's measured activity was about 0.005%
of active CPU. Background compilation was paused in BOTH after the original ten
warm-up updates, solely as a private diagnostic; no such policy ships.

| Independent pair | Measured wall time | CPU time | Instructions | CPU cycles |
| --- | ---: | ---: | ---: | ---: |
| 1, new relative to old | -1.64% | -1.09% | -2.48% | -1.29% |
| 2, new relative to old | -1.78% | -1.30% | -2.35% | -1.34% |

Every order block favoured the candidate, all losses matched the original reference,
and no sampled swap or guard event occurred. Four individually slower updates remain
recorded; correlated steps are not treated as independent experiments. These results
support removal of the sustained CPU regression on this workload, not a general
library speedup or a promise that every update is faster.

A subsequent matched-time control removes the compiler intervention entirely.
Two independent pairs use normal JVM compilation, the same full-size Loop model,
ten warm-up/thirty measured updates and original pre-ND baseline; initialization
order is reversed in the second pair. Only one trainer runs at a time, and CPU
activity in both JVMs, including background compilation while parked, is counted.
Measured wall time changes by -0.412%/-0.209%, active CPU by -0.973%/-0.391%,
instructions by -1.656%/-1.623%, and whole-window CPU by -0.926%/-0.306%.
All forty loss bits and data cursors per process match the original references.
These are affirmative normal-runtime observations, not proof of universal equality.
Positive shorter order blocks and main-thread/compiler costs remain visible;
more compiler CPU was not eliminated or renamed measurement noise. Six private
Java changes were rejected because lower compiler cost did not consistently lower
overall cost. That native-only stage did not alter those Java runtime bytes.

The later V37 normal-compilation Loop CPU64 control retains two independent pairs:
elapsed time -1.461%/-2.170%, active CPU -0.876%/-2.281%, and whole-window CPU
-0.650%/-2.096%. All forty loss bits and cursors per process match. Extra compiler
CPU and individually slower updates remain recorded. V38 changes only the GPU
buffer-binding class; these CPU results are explicitly V37 observations.

Nova Metal controls retained exact command signatures and native-memory boundaries.
Normal-runtime OS-thread accounting found extra background Java compilation in the
new runtime: 0.53–0.63 CPU seconds versus 0.34–0.37 over 30 measured updates. This
cost is real. With compilation separately controlled, both new noncompiler CPU
totals were below both old totals, while GPU duration changed with process order,
including identical old-code self-controls. There was no additional GPU command work.

More compiler CPU is not the same as slower elapsed startup. In the preserved
normal-compilation windows, all four matched Nova first-ten-update comparisons and
both Loop comparisons completed faster with the candidate. Whole-process timings
and individual first updates were mixed and are not relabelled as universally faster.
The evidence supports the tested training workflows; it does not establish identical
initialization latency, energy use or elapsed time under every machine condition.

## Remaining product limitations

FLOAT32 remains the default. Compact native matrix operands use FP32 accumulation;
other operations can use FP32 arithmetic on the selected backend. No implicit autocast,
loss scaling or hidden CPU fallback is enabled. Compact storage neither halves total
training memory nor guarantees faster training; the measured compact full-model paths
remain slower than the optimized FLOAT32 path on this Mac.

Public APIs and checkpoint formats remain alpha. See [tensors and precision](../../tensors.md),
[generation modes](../../inference.md) and [coding/regression requirements](../../quality-and-regression.md).
