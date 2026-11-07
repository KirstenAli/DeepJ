# Small Qwen: matched PyTorch training measurements

Measured on 7 November 2026 on an Apple M4 with 16 GiB unified memory,
macOS 15.5, Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
This is a randomly initialized **1,640,192-parameter Qwen3 configuration**,
not training of the pretrained Qwen3-1.7B checkpoint used by the inference table.
No production code changed for this comparison.

## Complete training steps

Lower milliseconds per step is better. Each process performs 100 warm-up training
updates followed by 30 timed updates on the same batch. Each table value is the
median of two fresh-process medians. The second pair reverses engine order.

| Backend | DeepJ (ms/step) | PyTorch (ms/step) |
| --- | ---: | ---: |
| CPU | 11.729 | **9.133** |
| Metal / MPS | **4.320** | 16.964 |

| Engine | First process median | Second process median |
| --- | ---: | ---: |
| DeepJ CPU | 11.695 ms | 11.763 ms |
| PyTorch CPU | 9.143 ms | 9.123 ms |
| DeepJ Metal | 4.318 ms | 4.323 ms |
| PyTorch MPS | 14.090 ms | 19.837 ms |

PyTorch CPU was faster for this training workload. DeepJ Metal had lower step
times than PyTorch MPS in both processes; the MPS process medians varied substantially.
These short measurements do not establish general training performance or a stable
GPU speedup ratio. The earlier DeepJ-only controls measured 11.500 ms on CPU and
3.962 ms on Metal; the README now uses these fresh matched-comparison measurements.

## Matched workload and timing

- FP32 parameters, activations, gradients and AdamW state; no autocast or quantization.
- Vocabulary 256, hidden width 256, two decoder blocks, four query heads,
  two KV heads, head dimension 128 and feed-forward width 512.
- RoPE theta 1,000,000, RMSNorm epsilon 1e-6, tied embedding/output weights,
  full causal attention and no dropout, gradient checkpointing or KV cache.
- Batch size one with 64 token IDs and 64 aligned target IDs. Targets are generated
  independently; both engines use mean cross-entropy on all 64 positions, without
  Transformers' optional internal next-token label shift.
- DeepJ initialization seed 2026, input seed 3 and target seed 4. DeepJ exports its
  initial weights and exact inputs; PyTorch loads every parameter from that export.
- AdamW learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8 and weight decay 0.01,
  applied to all parameters. Global L2 gradient clipping uses maximum norm one
  and coefficient `1 / (norm + 1e-12)` when clipping is needed.

Each timed step includes gradient clearing, forward, loss, backward, scalar loss
readback, global norm calculation/readback, clipping, AdamW and final synchronization.
DeepJ calls the existing `CausalLMTraining` trainer and
`releaseTemporaryResources()`, which flushes and waits for Metal work. PyTorch uses
Transformers' `Qwen3ForCausalLM` with SDPA attention, native autograd and AdamW,
`zero_grad(set_to_none=False)` and `torch.mps.synchronize()` after each GPU step.
The small clipping adapter matches DeepJ's epsilon and conditional scaling policy;
norm and scaling still use PyTorch tensor operations. PyTorch runs eagerly without
`torch.compile`, with MPS CPU fallback disabled.

Both engines use ten CPU threads; PyTorch has one inter-op thread.
`VECLIB_MAXIMUM_THREADS=10` is set for both. DeepJ uses `-Xmx1g -Xbatch
-XX:CompileThresholdScaling=0.1`. Engines run sequentially on the same machine,
outside the sandbox, through IntelliJ's terminal. Builds and correctness checks
finish before timing begins. Loading, initialization, checkpoint exports and
process startup are excluded. Warm-up updates are retained, so each process
finishes after 130 updates rather than resetting the model after warm-up.

## Numerical checks

Separate untimed checks compare initial logits, loss, raw gradients before clipping
and weights after one optimizer update. They use the same fixture as the timed runs.

| Check: maximum absolute difference | CPU vs PyTorch CPU | Metal vs PyTorch MPS |
| --- | ---: | ---: |
| Initial logits | 1.252e-6 | 1.311e-6 |
| Raw gradients | 2.608e-7 | 2.282e-7 |
| Weights after one update | 4.914e-6 | 1.482e-5 |
| Weights after 130 updates | 3.296e-5 | 1.464e-5 |
| Loss across 130 updates | 1.897e-4 | 1.765e-5 |

The training runs are numerically close, not bit-identical. DeepJ CPU's existing
global norm implementation accumulates squared gradients sequentially in FP32.
On the first step it reports 6.2244844, versus 6.2373562 from PyTorch's reduction
and approximately 6.2373656 from an FP64 sum of the captured raw gradients.
This produces about 0.207% more gradient scaling on that CPU step. The difference
is in the norm reduction; raw-gradient relative L2 error is about 1.23e-6.
Metal's first-step norm agrees with PyTorch MPS to displayed FP32 precision.
Optimizer operation order and FP32 rounding also differ between implementations.
No training implementation was altered to make the numbers agree.

After 130 updates, CPU loss is 0.1198061 in DeepJ and 0.1196944 in PyTorch;
Metal/MPS loss is 0.1196875 and 0.1196949 respectively. These checks establish
equivalent work for this bounded comparison, not identical long-run training.
Post-step gradient buffers are not used for comparison because DeepJ's temporary
buffer cleanup retires them; gradients are captured before clipping in the audit.

All 15 selected Qwen model/checkpoint tests passed. Checkstyle, PMD and duplication
checks passed for the benchmark additions.

## Reproduction and evidence

DeepJ production code is `4ed97933`; later commits through `3b79b30f` only change
documentation. [results.json](results.json) contains all measured step times,
loss sequences, initial inputs/configuration, numerical comparisons and SHA-256
hashes of the runtime JAR and exported initial weights.

Build that production revision with its native CPU/Metal libraries, then use the
benchmark sources from this revision. Use Python 3.11 with `torch==2.8.0`,
`transformers==4.51.3`, NumPy and safetensors. Set the following local paths:

```bash
export JAVA_HOME=/path/to/temurin-20.0.2/Contents/Home
JAR=/absolute/path/to/deepj.jar
PYTHON=/absolute/path/to/reference-env/bin/python
OUT=/absolute/path/to/new-benchmark-output
bash docs/benchmarks/qwen-training-2026-11-07/run.sh "$JAR" "$PYTHON" "$OUT" audit
bash docs/benchmarks/qwen-training-2026-11-07/run.sh "$JAR" "$PYTHON" "$OUT" measure
"$PYTHON" docs/benchmarks/qwen-training-2026-11-07/summarize.py "$OUT" "$JAR"
```

The output directory must be new. The launcher compiles the optional Java probes;
the audit creates the shared fixture and validates actual GPU availability. The
summary checks tensor names/shapes, finite outputs, initial-logit/gradient agreement
and the expected sample counts. Python is only a reference/testing dependency.
