# N-dimensional tensors: qualification summary

This report covers the source-branch ND migration, optional FLOAT16/BFLOAT16 storage,
native compact matrix paths and associated lifetime/performance fixes. It is not a
claim about the published `0.11.0-alpha` artifact or a universal speedup.

## Reproducibility and scope

Tests ran on an Apple M4 MacBook Air with 16 GB unified memory and Temurin 20.0.2.
Full-size models were Prism 90,128,896, Nova 99,711,552 and Loop 100,375,680 parameters.
Metal comparisons used 1,024-token contexts; explicitly bounded CPU comparisons
used 64 tokens. All families also have smaller independent operator/model tests.
Seed, data, optimizer, precision and cleanup policy were matched between versions.

Original pre-ND JAR SHA256:
`bfbadaf4c1fc3d3241fef5b81fdf611767d5f61926204b4e1021ea97c5a3d591`.
Final qualified runtime JAR SHA256:
`810da7396c277cc6be09c3aa84a5b60162f62fa3a5291c316429facfd9362ef8`.
Frozen source archive SHA256:
`06895eb7b1da83525e79f424b58dd62f3b3073c54195fb3499b923064e1b986e`.
The final source audit compared all 781 source/test/native/script/configuration
files with that archive; all bytes matched. Only documentation changed afterward.

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

The final local full-coverage build ran 2,260 tests: 2,248 passed, zero failures or
errors, and 12 explicit skips. Eleven are disabled manual timing tests; one is an
optional local-checkpoint smoke test requiring `deepj.modelPath`. Actual Metal,
native instrumentation, desktop/frontend, quality and coverage gates passed.

## Defects found and corrected

Qualification exposed real defects rather than just measurement variation:
scoped-view ownership, unnecessary device copying, dense ADD dispatch, checkpoint
initialization/staging and evaluation-output lifetimes. Regression tests now cover
these paths. CPU migration also introduced serial dense pointwise/SwiGLU work and
checked geometry/storage access inside inner loops. Parallel dispatch was restored,
invariant checks were hoisted, and avoidable shape/broadcast metadata was removed.
No numerical tolerance, production forced-GC policy or JVM compiler workaround was
added to make a result pass.

Early native peak differences were traced separately. Logical buffer requests and
retained memory matched; weak-reference-dependent scratch eligibility already
existed in the old runtime. A private control of that eligibility reproduced equal
physical allocation/retirement totals. Historical swap/peak guard failures remain
recorded failures. System-wide swap observations are not attributed to one process
without process-specific evidence.

## Performance investigation

Original uninstrumented old/new timings and reversed/self controls are preserved.
Small positive timings were not waived using an arbitrary percentage margin.
Supplemental diagnostics isolated operation costs, allocation, native commands,
actual GPU time, main/process CPU work, compilation and process-order variation.
The diagnostics below explain the remaining gaps; they do not replace those
original workloads or justify a universal speed guarantee.

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
