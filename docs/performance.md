# Inference and training benchmarks

In **Apple M4 tests**, DeepJ achieved higher **Qwen3-1.7B decode throughput** across
**FP32, FP16 and BF16**, plus lower **training step times** for the small Qwen and Gemma configurations
in FP32 and mixed precision below.

| Measurement | Precision | DeepJ | PyTorch |
|---|---|---:|---:|
| CPU cached decoding | FP32 | **12.72 tokens/s** | 9.02 tokens/s |
| CPU cached decoding | FP16 | **21.60 tokens/s** | 20.93 tokens/s |
| CPU cached decoding | BF16 | **22.29 tokens/s** | 20.09 tokens/s |
| CPU prompt processing | FP32 | **283.7 ms** | 294.0 ms |
| CPU prompt processing | FP16 | **322.3 ms** | 707.9 ms |
| CPU prompt processing | BF16 | **725.9 ms** | 1028.6 ms |
| Metal / MPS cached decoding | FP32 | **12.69 tokens/s** | 12.05 tokens/s |
| Metal / MPS cached decoding | FP16 | **24.00 tokens/s** | 23.48 tokens/s |
| Metal / MPS cached decoding | BF16 | **24.02 tokens/s** | 23.63 tokens/s |
| Metal / MPS prompt processing | FP32 | **119.7 ms** | 132.3 ms |
| Metal / MPS prompt processing | FP16 | **104.7 ms** | 106.1 ms |
| Metal / MPS prompt processing | BF16 | **104.2 ms** | 105.6 ms |
| Metal / MPS streaming chat | FP16 | **22.93 tokens/s** | 22.40 tokens/s |
| Metal / MPS streaming chat | BF16 | **23.08 tokens/s** | 22.61 tokens/s |
| CPU training step (small Qwen) | FP32 | **6.718 ms/step** | 9.016 ms/step |
| CPU training step (small Gemma) | FP32 | **7.163 ms/step** | 7.656 ms/step |
| Metal / MPS training step (small Qwen) | FP32 | **4.113 ms/step** | 17.824 ms/step |
| Metal / MPS training step (small Gemma) | FP32 | **10.792 ms/step** | 18.072 ms/step |
| CPU training step (small Qwen) | FP16 (mixed) | **10.709 ms/step** | 81.029 ms/step |
| CPU training step (small Qwen) | BF16 (mixed) | **10.925 ms/step** | 81.784 ms/step |
| CPU training step (small Gemma) | FP16 (mixed) | **19.198 ms/step** | 60.181 ms/step |
| CPU training step (small Gemma) | BF16 (mixed) | **26.569 ms/step** | 60.583 ms/step |
| Metal / MPS training step (small Qwen) | FP16 (mixed) | **4.937 ms/step** | 18.008 ms/step |
| Metal / MPS training step (small Qwen) | BF16 (mixed) | **5.124 ms/step** | 17.484 ms/step |
| Metal / MPS training step (small Gemma) | FP16 (mixed) | **6.823 ms/step** | 19.228 ms/step |
| Metal / MPS training step (small Gemma) | BF16 (mixed) | **6.864 ms/step** | 16.324 ms/step |

FP16/BF16 training rows compare DeepJ compact training with PyTorch AMP, using
FP32 gradients and optimizer state. [Training precision policies and measurements](benchmarks/compact-training-2026-11-08/REPORT.md).

## Setup and measurements

Inference uses the pretrained Qwen3-1.7B checkpoint on an Apple M4 with 16 GiB
unified memory and macOS 15.5. Cached-decoding and prompt-processing tests use
batch size one, a 64-token prompt, a KV cache and 16 fixed continuation tokens,
with full final-position logit readback and no sampling. Streaming-chat tests
generate and render their replies. Higher tokens/s and lower milliseconds are better.

Training uses separate small, randomly initialized Qwen and Gemma configurations,
using FP32 or the mixed-precision policies linked above. Complete steps include forward, loss,
backward, gradient clipping, AdamW and synchronization. These are separate runs
from the full-checkpoint inference measurements.

The reports record each configuration, measured revision, raw trials and numerical checks:

- [FP32 Qwen3-1.7B inference](benchmarks/full-qwen-2026-11-07/REPORT.md).
- [Updated FP32 CPU prompt processing](benchmarks/full-cpu-prompt-2026-11-07/REPORT.md).
- [FP16/BF16 CPU inference](benchmarks/compact-followup-2026-11-07/REPORT.md).
- [FP16/BF16 Metal decoding and streaming chat](benchmarks/demo-compact-2026-11-07/REPORT.md).
- [Small Qwen training](benchmarks/compact-cpu-metal-2026-11-07/REPORT.md#fp32-training-and-regression-controls).
- [Small Gemma training on CPU](benchmarks/gemma-cpu-indexing-2026-11-07/REPORT.md) and
  [Metal](benchmarks/gemma-video-gap-2026-11-07/REPORT.md#small-gemma-training-comparison).

## Image and video generation

Complete staged generation on Apple M4 Metal/MPS, including stage loading.
Lower seconds means faster generation. Photos use two fresh-process runs per
engine and precision; videos use one short preview per engine and precision.

| Workload | Precision | DeepJ Metal | PyTorch / Diffusers MPS |
|---|---|---:|---:|
| SANA-Sprint image, 1024×1024, 2 steps | FP32 | **18.56 s** | 20.03 s |
| SANA-Sprint image, 1024×1024, 2 steps | FP16 | **17.51 s** | 18.38 s |
| SANA-Sprint image, 1024×1024, 2 steps | BF16 | **16.51 s** | 17.14 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | FP32 | **14.62 s** | 23.59 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | FP16¹ | **12.87 s** | 24.88 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | BF16 | **13.74 s** | 14.64 s |

¹ FP16 video uses the same range-preserving FP32 projection, attention-denominator
and residual policy in both engines. Python FP32 video streams weights to fit
the 16 GiB machine. See the [setup, raw trials and numerical checks](benchmarks/media-throughput-2026-11-07/REPORT.md)
for the staged Python reference, precision settings and timing scope.
