# FP32 Metal implementation of FlashAttention: qualification

24 October 2026. All declared acceptance checks passed. This is a bounded
qualification, not proof of month-long convergence or identical old/new weights.

## Implementation and scope

Baseline: `d879b55c2c1814465f5224566f5d85777a4d2488`, the shipped parallel blocked
MPS attention implementation. Candidate: `1916ef4230c4e3392b16cae219d584f59c50be94`.
Later evidence-only commits do not change the measured implementation.

The replacement uses FP32 online softmax, SIMD-group matrix operations and
parallel query blocks. Backward assigns separate ownership to query gradients
and key/value gradients, avoiding atomics. It removes the old attention scratch
buffers and multi-queue/event orchestration. No mixed precision, optimizer,
training schedule, public model API or checkpoint format change is included.

Automatic selection retains the existing shape restrictions: 64-wide heads,
matching query/key/value shapes, finite scale, and sequence lengths of at least
256 divisible by 32. Unsupported cases use the ordinary Metal attention path.
All five architectures share this routing. CPU behavior is unchanged.

Capability detection follows successful attention pipeline creation. Hosted
Apple Paravirtual hardware reports Apple7=false but successfully executed a
dynamic-input SIMD matrix probe (64 outputs, maximum error zero). The original
family-only gate incorrectly skipped supported tests. That was fixed without
changing shader arithmetic or lowering coverage thresholds. Temporary diagnostic
workflows were removed from the final branch.

## Matched training-step timings

Apple M4, 16 GiB unified memory, JDK 20, batch 1, context 1024. Three fresh JVMs
per variant/model, alternating old/new order, 10 warmup and 30 measured updates
per JVM. Reported time is the median of the three JVM medians. Synthetic tokens
are used here; independent real-data validation is described below.

| Model | Parameters | Baseline seconds/step | Candidate seconds/step | Less step time | More throughput | Lower peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prism | 90,128,896 | 0.745052 | 0.604525 | 18.86% | 23.25% | 1.40% |
| Nova | 99,711,552 | 1.277007 | 1.033931 | 19.03% | 23.51% | 2.23% |
| Loop | 100,375,680 | 2.100160 | 1.564771 | 25.49% | 34.22% | 2.33% |

Every matched timing pair improved. All 18 cases completed successfully and
recorded zero new system-wide swapout pages. Time reduction is `1 - new/old`;
throughput gain is `old/new - 1`. These are machine/session-specific measurements,
not universal speed guarantees. The full sample values, RSS and OS peak footprint
are in [timing-summary.json](timing-summary.json).

The memory metric is whole-process peak footprint from `/usr/bin/time -l`, not
GPU-only allocation or RSS. The baseline already avoided retaining full attention
matrices. We therefore measured modest additional total-memory savings, not the
large naive-attention-to-Flash reduction. Parameters, gradients, optimizer state
and other Java/native allocations remain. RSS varies independently and is retained
in the evidence rather than being substituted for footprint.

## Real-data and numerical qualification

Thirty fresh runs completed: five architectures, initialization seeds 100/101/102,
baseline and candidate, 200 updates each. This is 6,000 optimizer updates and 150
held-out measurements, with identical precomputed batches within each old/new pair.
Separate TinyStories training/validation files use the Frisky BPE tokenizer with
vocabulary 16,384. Validation uses a fixed 8,192-token set at steps 0/50/100/150/200.
AdamW learning rate is 1e-4. No existing model weights or training checkpoints
were loaded or overwritten.

Origin (4,605,952 parameters) and Orbit (4,540,288) use bounded two-layer fixtures.
Prism, Nova and Loop use the full configurations listed above. Loop has 6 prelude,
12 shared core and 6 coda layers, with two core passes: 36 executed layers.
Prism uses a 6 GiB Java heap; the other cases use an 8 GiB maximum.

| Measurement across all pairs | Largest absolute difference | Declared acceptance bound |
| --- | ---: | ---: |
| Held-out loss | 0.00000557303 nats | 0.01 nats |
| Mean training loss over 200 updates | 0.00000066755 nats | 0.01 nats |
| Individual training-step loss | 0.00001216 nats | Recorded and reviewed for drift |

All sampled gradient norms were finite and positive. Every individual loss delta
and each 50-update window were inspected. Loop seed 101 showed a small increase
in late differences; its final-window mean delta was -0.0000020078 nats. This
remained far inside the declared bounds, not an exploding or practically large
loss divergence. [training-summary.json](training-summary.json) preserves every
per-step delta; [audit.json](audit.json) includes the window and memory review.

The full build also passed:

- Independent scalar FP64 attention checks for forward and dQ/dK/dV, including
  production 9/12-head, 1024-token cases, causal boundaries and saturated scores.
- Existing per-element bounds: output `2e-5 + 2e-4*abs(reference)`; derivatives
  `3e-5 + 5e-4*abs(reference)`. These were not increased.
- Finite differences for both the independent derivative reference (108 checks)
  and GPU backward implementation; zero/negative scale and fallback coverage.
- Bit-identical same-implementation repeated outputs, gradients and three-step
  training runs across Origin, Orbit, Prism, Nova and Loop.
- Cross-algorithm whole-model comparison: loss bound
  `2e-4 + 2e-4*abs(reference)`, global gradient relative L2 <=1e-3, weight relative
  L2 <=1e-4. Final-build maxima were 5.821e-6 for gradients and 3.388e-7 for weights.

### Why the previous strict weight checks changed

The independent investigation traced selected differences to FP32 reduction order,
cancellation in near-zero gradients and AdamW amplification. Substituting FP64
attention into the otherwise FP32 models also failed the old selected-weight gate.
All 56 selected optimizer replays reproduced their recorded updates bit for bit.
No optimizer epsilon, clipping or learning-rate change was used to hide differences.

The cross-algorithm exact/legacy-trajectory assertions were deliberately replaced
by the independently declared accuracy, whole-model and training checks above.
Same-implementation exact-repeat checks remain. This contract change is explicit;
it is not a claim that old and new weights match bit for bit, that Flash is always
more accurate, or that all possible inputs have been tested. The archived
`investigation/REPORT.md` explains the controls and cancellation mechanism.

## Memory, GC and limitations

Heap minima/maxima and GC time were reviewed in four 50-update windows per run.
The full-size models continued reclaiming memory within their fixed heap limits;
no OOM or clear accumulating-retention pattern was observed during these 200-step
checks. Sampling at step boundaries is not a heap-dump or long-duration leak proof.

Unlike the dedicated timings, five real-data runs recorded new system-wide
swapouts: Nova seed100 baseline 47,396 pages; Loop seed100 baseline 115,340;
Loop seed101 baseline 19,760 and candidate 7,176; Loop seed102 baseline 4,260.
That is five runs, totaling 193,932 pages of 16 KiB. These counters cover the
whole machine and cannot be attributed entirely to one process. Real-data run
times are not used as the speed headline. Other runs recorded zero new swapouts.

The CPU/Metal reference tests and bounded training support this implementation's
acceptance. They do not guarantee identical generated text, identical long-run
weights, improved model quality, or performance on unmeasured hardware/shapes.

## Build and provenance

Initial and final physical-Mac `mvn -o clean verify -Pfull-coverage` both passed:
978 Java tests, zero failures/errors, 12 expected skips. Eleven are deliberately
disabled manual all-ops timing tests; one needs an external Prism checkpoint.
Metal correctness, Flash and JavaFX tests actually ran. Frontend: 43 tests with
100% measured coverage. Native quality: 32 parser tests and zero function-rule
violations (135 host, 24 attention shader, 85 other shader functions).

Final coverage: full Java 91.51% lines/78.84% branches; Metal Java 98.31%/94.08%;
autograd 100%/98.15%; native host 97.15%/87.50%, 100% functions. GPU kernel entry
selection is 100%; this is not shader-line or shader-branch coverage.
All configured thresholds and style/complexity/nesting/duplicate checks passed.

[Full hosted Mac CI at the measured source revision passed](https://github.com/KirstenAli/DeepJ/actions/runs/40101748932).
Every tracked file in the frozen baseline/candidate snapshots was checked against
its corresponding Git revision. Bundled library SHA-256:

```text
baseline  785b790a0d07e896ab78d67205f721b68e63aac991ecf97b0066b48b66fa0029
candidate 259bc5217944808414cc25f9f6e49f6d571187a0215036e44d9daf071d055cc7
```

## Evidence and reproduction

[evidence.tar.gz](evidence.tar.gz) contains the predeclared plan, all 30 real-data
and 18 timing logs with before/after VM counters, both full-build logs, aggregate
statistics, harness/analyzer/runner sources, earlier numerical investigation and
hosted capability diagnostics. It contains no datasets, token batch binaries,
tokenizer, model weights, private checkpoints or credentials. Local home paths
are replaced with `<LOCAL_HOME>`; original/sanitized file hashes are in the manifest.

Archive SHA-256:
`c0ccf5532a7d5cbb31e63127a57701ab35c1f4c3ec24dbfee4d496f0ad5cbeb1`.

To reproduce, create separate Git archive snapshots at the two revisions, compile
their production classes with JDK 20, and place them under `baseline/` and
`candidate/` in a new experiment directory. Copy `sources/*` there and create empty
`logs/` and `harness/` directories. Supply the separate TinyStories files and Frisky
BPE tokenizer expected by `QualificationData`, and replace `<LOCAL_HOME>` in the
runner with the corresponding local path. With other GPU jobs stopped, run
`caffeinate -i /bin/zsh run-all.zsh` on a supported Mac. It refuses to overwrite case
logs or precomputed token batches. Requires Maven, cached dependencies for offline
builds, and a desktop session for JavaFX tests. Do not point the experiment's output
at an existing real training checkpoint directory.
