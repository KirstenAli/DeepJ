# Gemma CPU training: shared indexing improvement

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5;
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
Baseline: `b89c4d48` on `feature/shared-tensor-performance`.

## Matched results

The **1,247,488-parameter FP32 Gemma** configuration and complete training step
are unchanged from the [preceding CPU comparison](../gemma-cpu-followup-2026-11-07/REPORT.md):
two layers, width 256, MLP width 512, four query heads, two KV heads,
head width 64 and vocabulary 256. Both engines load the same checkpoint and
use the same next-token batch: one sequence with 64 input tokens and 64 targets.

Each process performs **100 warm-up updates and 30 timed updates**. Each engine
runs in **six fresh processes**. The six rounds cover every ordering of
baseline DeepJ, updated DeepJ and PyTorch exactly once, avoiding a consistent
first/last advantage. Each table value summarizes the six process medians;
all 180 timed steps per engine remain in [results.json](results.json).

| Engine / version | Median of process medians (ms/step) | Process median range (ms/step) |
| --- | ---: | ---: |
| DeepJ baseline | 7.666 | 7.625–7.744 |
| DeepJ with row indexing | **7.163** | 7.132–7.193 |
| PyTorch | 7.656 | 7.516–7.766 |

DeepJ's updated step time is **6.4% lower than PyTorch's** and **6.6% lower than
its own baseline**. It is lower than PyTorch in **all six run orders**.
This is a result for the specified small FP32 CPU training configuration,
not a full-size Gemma training measurement or a claim about every PyTorch workload.
Absolute timings vary across benchmark batches; the before/after/reference
measurements in this table were collected together.

The timed step includes zeroing gradients, forward, mean cross-entropy,
backward, global gradient clipping at norm 1 and AdamW. Learning rate is 0.0001,
betas are 0.9/0.999, optimizer epsilon is 0.00000001 and weight decay is 0.01.
PyTorch uses built-in `clip_grad_norm_` with `error_if_nonfinite=True` and its
default `foreach`/`fused` optimizer settings. Both engines check nonfinite loss
and gradient norm. DeepJ's clipping epsilon remains 0.000000000001;
PyTorch's is 0.000001. Both use ten CPU threads; PyTorch uses one inter-op thread.
The reference uses eager attention to retain Gemma's attention softcap.

Loading and process startup are outside step timing. IntelliJ is closed during
the measurements. The previously authorized photo/Spotlight jobs are paused,
then resumed after the batch. Each guarded process completed normally, with
no system swap-out counter growth; peak process RSS was below 350 MiB.

## What changed

The CPU profile identified repeated per-value coordinate decoding in indexed
gathers and gradient accumulation. Gemma exercises these operations when
repeating KV heads and propagating gradients through strided views.

The shared `CpuRowSelection` implementation now:

- Resolves source coordinates once per row when selecting a middle axis from
  a tensor whose final axis is contiguous, then copies the row's storage bits.
- Resolves source and destination row coordinates once for FLOAT32 indexed
  updates, then adds values using their physical strides. Repeated indices
  retain the original logical update order.

The existing contiguous, last-axis gather, compact accumulation and mixed-dtype
fallbacks are retained. This is a general CPU tensor improvement; it adds no
Gemma-specific branch and changes no network architecture, loss, optimizer,
thread-pool setting, native library, Metal source or autograd implementation.

A separate routing experiment for contiguous last-axis updates did not improve
step time and was not promoted.

## Correctness

The new tests cover all storage dtypes, middle/last axes, offsets, permutations,
nonunit strides, aliases, empty selections, repeated-index cancellation, compact
updates into FLOAT32 targets, and raw signed-zero/NaN storage bits.

An instrumented, ten-step Gemma control checks every optimized operation against
the original per-element implementation **in the same execution**: **20 gathers
and 20 indexed additions matched exactly**. This diagnostic changes timing and
is excluded from the benchmark table.

Ten-step Qwen, Loop and Prism controls, using width 64, vocabulary 128, batch
size one, 32-token sequences and one executor worker, retain identical loss
sequences and final parameter/gradient hashes on **both CPU and Metal**.

The complete small Gemma training runs start at loss 5.5921097 and finish near
0.24936. Across the six paired native CPU runs, the maximum loss-sequence
before/after difference is 0.0000811; the largest final parameter difference is
0.0000218. The maximum loss difference against PyTorch is 0.0000547.
Independent baseline runs also have small numerical variation; full Gemma
training states are not claimed to be bit-identical between fresh processes.
Separate serial/native and Java-only controls are retained in the results,
alongside the exact per-operation oracle above.

## Build and reproduction

`mvn -o verify -Pcli -Dgemma.test.metal=true` passed in the authorized IntelliJ
terminal with physical Metal: **3,010 tests, zero failures/errors, 57 optional
skips**. Checkstyle, PMD, duplication, frontend and native quality gates pass.
Core Java line/branch coverage is 96.21%/85.91%; autograd is 99.65%/96.09%.
An additional Java-only fallback run passed all 27 selected indexing tests.

[Evidence](evidence.tar.gz) contains the probes, fixture generator, launch script,
every timing/loss sample, resource metrics, regression controls, indexing oracle
and build logs. Generate the small fixture with
`gemma_training.py <fixture-directory> fixture`, compile the Java probes against
the appropriate CLI jars and replay `balanced-comparison.sh` after substituting
local paths. `results.json` records checkpoint, source and jar hashes.
No pretrained image/video generation is part of this check.
