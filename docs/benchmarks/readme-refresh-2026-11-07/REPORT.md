# README benchmark verification

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5;
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
Tested source: `6d1c1342d1a63b392253917ea0ac9353f82e3731`.
This update changes documentation only.

## Small Gemma Metal training

Both engines load the same **1,247,488-parameter FP32 Gemma** checkpoint:
two layers, width 256, MLP width 512, four query heads, two KV heads,
head width 64 and vocabulary 256. Each update uses the same next-token batch:
one sequence with 64 input tokens and 64 targets. This is the fixture used in
the [Gemma CPU comparison](../gemma-cpu-indexing-2026-11-07/REPORT.md).

Six fresh processes per engine each perform **100 warm-up updates and 30 timed
updates**. Engine order alternates between pairs. Each README value is the
median of the six process medians, without selecting or discarding any
completed training process.

| Pair | Order | DeepJ Metal (ms/step) | PyTorch MPS (ms/step) |
| --- | --- | ---: | ---: |
| 1 | DeepJ, PyTorch | 10.682 | 21.740 |
| 2 | PyTorch, DeepJ | 9.817 | 20.534 |
| 3 | DeepJ, PyTorch | 10.336 | 20.595 |
| 4 | PyTorch, DeepJ | 12.803 | 9.728 |
| 5 | DeepJ, PyTorch | 12.676 | 15.611 |
| 6 | PyTorch, DeepJ | 10.903 | 12.617 |
| **Median of process medians** | | **10.792** | **18.072** |

DeepJ's aggregate step time is **40.3% lower** in this batch. It is lower in
five of six pairs; PyTorch is lower in pair four. Both engines show substantial
run-to-run variation, so this result does not establish a fixed speedup or
performance for full-size Gemma training.

Each timed step includes zeroing gradients, forward, mean cross-entropy,
backward, checking finite loss and gradient norm, global gradient clipping at
norm 1, AdamW and synchronization. Learning rate is 0.0001, betas are 0.9/0.999,
optimizer epsilon is 0.00000001 and weight decay is 0.01. PyTorch uses built-in
`clip_grad_norm_` with `error_if_nonfinite=True`, default `foreach`/`fused`
optimizer settings and eager attention to preserve Gemma's attention softcap.
DeepJ's clipping epsilon is 0.000000000001; PyTorch's is 0.000001.
Both use ten CPU threads; PyTorch uses one inter-op thread.

Loading and startup are outside step timing. Both engines start at loss
5.5921097 and finish near 0.2493. The maximum absolute difference across paired
130-update loss sequences is 0.000103; fresh-process training states are not
claimed to be bit-identical. Every sample is finite. All twelve guarded
training processes completed normally with no new system swap-out and peak
process RSS below 552 MiB. Physical Metal is required by the Java probe;
PyTorch MPS CPU fallback is disabled. IntelliJ is closed during measurements.
The previously authorized background jobs are paused and resumed by a trap,
with an independent bounded resume watchdog.

## CPU prompt follow-up

The README then retained the preceding matched FP32 CPU prompt comparison:
**292.8 ms for DeepJ and 286.5 ms for PyTorch**. That approximately 2.2% gap
had not been verified as closed. Those historical measurements and their
protocol are in the [full Qwen report](../full-qwen-2026-11-07/REPORT.md).

A later [completed full-model CPU comparison](../full-cpu-prompt-2026-11-07/REPORT.md)
measures 283.7 ms for DeepJ and 294.0 ms for PyTorch and replaces that README row.

A current prompt-only comparison was attempted using the same pretrained
Qwen3-1.7B checkpoint and 64 prompt token IDs, FP32 weights, a KV cache and
full final-position logit readback. The plan was two fresh processes per engine,
reversed order between pairs, eight warm-ups and five timed trials per process.
Neither engine's startup or weight loading belongs inside prompt timing.

The initial DeepJ process exceeded the 64 MiB system swap-out growth guard.
A retry with Chrome closed and a smaller 8 GiB Java heap failed during weight
loading with `OutOfMemoryError`. Restoring the original 10 GiB heap with Chrome
and IntelliJ closed still triggered the swap guard during warm-up. The guard
terminated these processes; the launcher resumed the background jobs after
each attempt. No process reached measured trials, and the PyTorch processes
were consequently not launched. These attempts provide **no accepted current
CPU performance comparison** and are excluded from README values. They do not
identify the cause of memory pressure or establish an execution-time regression.

The attempted prompt-only protocol also omits the 16 decode steps between
prompts used by the historical benchmark, so it would be a new paired
comparison rather than a controlled before/after optimization measurement.
No cached-decoding or other README rows were remeasured in this update.

## Evidence and reproduction

[results.json](results.json) retains all training times, loss sequences, resource
metrics, hashes and excluded CPU attempt details. The tested CLI JAR SHA-256 is
`9719491b85ed5b22ebf4522e0982703daba977fbc94b80aacb73e9e60aab54b6`.

[evidence.tar.gz](evidence.tar.gz) contains both training probes, the fixture
generator and configuration, launch scripts, raw logs, guards, prompt input and
the optional Qwen benchmark sources. Model weights and application binaries are
excluded. Generate the small fixture using
`gemma_training.py <fixture-directory> fixture`, compile `GemmaTrainingProbe.java`
against the recorded CLI build, and substitute local paths in the launcher.
The initial launcher runs training before the guarded full-model prompt attempt;
omit that prompt section to reproduce only the small training comparison.
The Python files are comparison tools and are not part of DeepJ's runtime.

Documentation checks verify the README values against all 360 timed training
samples, aggregation, paired ordering, resource guards, table columns, local
report links and archive contents. `git diff --check` passes. No production,
test, native, UI or build source is modified, so the full application suite is
not rerun for this documentation update.
