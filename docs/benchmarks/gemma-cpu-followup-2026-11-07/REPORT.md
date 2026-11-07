# Gemma CPU training follow-up

7 November 2026, after `d4e0abbb` on `feature/shared-tensor-performance`.
Apple M4, 16 GiB unified memory, macOS 15.5; Temurin 20.0.2,
PyTorch 2.8.0 and Transformers 4.51.3.

## Matched CPU results

The same **1,247,488-parameter FP32 Gemma** checkpoint, input batch and complete
training step are used as in the [preceding comparison](../gemma-video-gap-2026-11-07/REPORT.md#small-gemma-training-comparison):
two layers, width 256, MLP width 512, four query heads, two KV heads, head width
64 and vocabulary 256. Batch size is one, with 64 input tokens and 64 targets.

Timing includes zeroing gradients, forward, mean cross-entropy, backward,
global gradient clipping at norm 1 and AdamW. Learning rate is 0.0001,
betas are 0.9/0.999, optimizer epsilon is 0.00000001 and weight decay is 0.01.
PyTorch uses its built-in `clip_grad_norm_` with `error_if_nonfinite=True` and
its default optimizer `foreach`/`fused` settings. Both check for nonfinite loss
and gradient norm. The previously documented clipping epsilon difference remains.
Both use ten CPU threads; PyTorch uses one inter-op thread.

Each process performs **100 warm-up updates and 30 timed updates**. Two fresh
processes per version/engine are run in this order: baseline, candidate, PyTorch;
then PyTorch, candidate, baseline. IntelliJ is closed and the authorized
background jobs are temporarily paused, then resumed. No benchmark triggered
swap growth; peak process RSS was below 350 MiB. Model loading is outside the
step timings.

| Version / engine | First process (ms/step) | Second process (ms/step) | Median of process medians (ms/step) |
| --- | ---: | ---: | ---: |
| DeepJ before this change | 9.599 | 9.395 | 9.497 |
| DeepJ with this change | 7.659 | 7.646 | 7.653 |
| PyTorch | 7.305 | 7.463 | 7.384 |

DeepJ step time fell **19.4%**. It remains **3.6% higher** than PyTorch in this
short test. This is an improvement toward parity, not a claim that DeepJ wins
this workload or that the remaining difference is statistically significant.
Every timed step and loss is retained in [results.json](results.json).

A [subsequent shared indexing improvement](../gemma-cpu-indexing-2026-11-07/REPORT.md)
measures DeepJ below PyTorch in all six balanced process orders.

## Shared CPU implementation

- FLOAT32 matrix addition reads transposed inputs in 64-by-64 tiles and uses
  four-by-four NEON transposes/additions. This accelerates parameter-gradient
  accumulation without per-value layout division. Offsets, padded rows, odd
  dimensions and overlapping views preserve logical addition. Differently
  mapped aliases are snapshotted before writing.
- Contiguous FLOAT32 AdamW updates process four independent values at a time.
  Floating-point contraction is disabled: the original Java expression order,
  divisions and decay update are retained. The Java and strided fallbacks remain.
- Large dense FLOAT32 gradient norms use 1,024-value blocks with vector
  sum-of-squares and double-precision accumulation of block totals. This avoids
  one long serial FLOAT32 accumulator. **The reduction order changes on CPU**;
  the loss, optimizer settings and clipping rule do not. Views and compact
  tensors retain their logical-layout fallback.

The optimizer matches the original scalar implementation bit-for-bit in a
check of over 1.3 million element updates, as well as the focused tests for offsets,
non-vector tails, zero/nonzero decay and multiple updates. For a separate
1,247,488-value norm check, relative squared-norm error against a double-precision
reference decreased from about 0.00050 to 0.000000036.

The new Gemma loss sequences differ from PyTorch by at most **0.000020** over
130 updates, and all runs finish near loss 0.24936. Relative to the old DeepJ
runs, the maximum loss difference is 0.000358 and the maximum final parameter
change is 0.000088. The norm change is therefore not claimed to preserve CPU
training hashes. Ten-step CPU Qwen/Loop/Prism controls differ in loss by at most
0.0000006. Those controls use width 64, vocabulary 128, batch size one, 32-token
sequences and one executor worker; their exact configurations are in the probes.

## Metal timing variation and the word “pooled”

“Pooled” in the preceding report means combining timing samples. It does not
refer to a thread pool, allocator pool or a newly discovered runtime fault.
The earlier separate Metal process medians are retained there because a pooled
median obscures their variation.

A new diagnostic run, with no Metal code changes, measured standard complete-step
medians of **9.520 and 11.474 ms** for DeepJ and **17.914 and 17.954 ms** for
PyTorch. Extra synchronized PyTorch phase diagnostics measured roughly 5.5 ms
forward/loss, 7.2 ms backward, 5.0–5.1 ms loss readback/clipping and 5.9–6.0 ms
optimizer update. Those extra barriers change the workload; these phase timings
are diagnostic and must not be compared with the standard step totals.

Variation alone does not identify a software defect. No DeepJ thread-pool fault
was reproduced. Raising the CPU serial-work thresholds made the small training
step slower, so those settings were left unchanged. A native softmax-backward
prototype also regressed the step and was not promoted.

**This change modifies no Metal source or Metal arithmetic.** Qwen, Loop and
Prism Metal controls produced identical loss sequences and final
parameter/gradient hashes before and after it.

## Status of 16-bit inference

The preceding [Qwen compact comparison](../demo-compact-2026-11-07/REPORT.md)
measured Metal FP16/BF16 decode at 24.00/24.02 tokens/s versus PyTorch's
23.48/23.63 for the fixed continuation workload. The
[CPU compact comparison](../compact-followup-2026-11-07/REPORT.md) also measured
faster Qwen FP16/BF16 prompt processing and cached decoding. These are the
measured Qwen inference gaps that were closed; these CPU training changes do
not constitute a new 16-bit inference benchmark.

The full Gemma inference and tiny video gaps in the
[preceding report](../gemma-video-gap-2026-11-07/REPORT.md) remain. Qwen and Gemma
share DeepJ tensors, autograd and backends, but use different operations,
normalization, activations, attention policies and dimensions. Gemma's earlier
large training regression included generic implementation routes that were
fixed in the preceding change; the work here improves shared CPU operations.

## Validation and reproduction

`mvn -o verify -Pcli -Dgemma.test.metal=true` passed in the authorized IntelliJ
terminal with physical Metal: **2,998 tests, zero failures/errors, 57 optional
skips**. Checkstyle, PMD, duplication, frontend and native quality gates pass.
Core Java line/branch coverage is 96.20%/85.85%; autograd is 99.65%/96.09%.

The final native loops also avoid counter overflow near the maximum tensor
size. After that bounds hardening, the CPU package check passed all 32 focused
tests and repeated the arithmetic/model smoke checks. A separate Java-only run
passed 18 tests with four native-only accuracy cases skipped.

[Evidence archive](evidence.tar.gz) includes probes, the fixture generator,
raw timing/loss logs, controls, arithmetic checks and build output. Generate the
small fixture with `gemma_training.py <fixture-directory> fixture`, compile the
Java probes against each CLI jar, and run the commands retained in
`final-checks.sh`. The tests and benchmarks use the small fixture; no full
image/video generation is performed in this follow-up. Jar/native/checkpoint
identities are retained in `results.json`.
