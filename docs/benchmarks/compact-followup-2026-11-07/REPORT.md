# Shared compact execution on CPU and Metal

Measured on 7 November 2026 on an Apple M4 with 16 GiB unified memory,
macOS 15.5, Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
This follow-up starts at `5e9156d6` on `feature/shared-tensor-performance`.
The optimizations apply to shared tensor operations and backend execution.

## Full Qwen3-1.7B inference

| Backend | Dtype | DeepJ prompt (ms) | PyTorch prompt (ms) | DeepJ decode (tokens/s) | PyTorch decode (tokens/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | FP16 | 322.3 | 707.9 | 21.60 | 20.93 |
| CPU | BF16 | 725.9 | 1028.6 | 22.29 | 20.09 |
| Metal / MPS | FP16 | 106.3 | 105.4 | 23.28 | 23.05 |
| Metal / MPS | BF16 | 105.3 | 105.8 | 23.53 | 23.61 |

CPU prompt processing and decoding were faster in both compact formats in this
workload. Metal prompt processing and decoding were within 1% in both formats
in the summarized measurements. This describes the measured workload, without
claiming statistical equivalence or performance across all models.

Each process uses five warm-up trials followed by five measured trials. CPU
values are process medians from one fresh process per engine and dtype. Metal
values summarize two fresh processes per engine and dtype by taking the median
of their process medians; the second pair reverses engine order.

| Metal dtype | Process pair | DeepJ decode (tokens/s) | PyTorch decode (tokens/s) |
| --- | --- | ---: | ---: |
| FP16 | 1 | 23.34 | 22.48 |
| FP16 | 2 | 23.22 | 23.62 |
| BF16 | 1 | 23.57 | 23.75 |
| BF16 | 2 | 23.50 | 23.47 |

The batch size is one, the prompt has 64 tokens, and both engines process the
same 16 fixed continuation tokens using a KV cache, without sampling. Each
forward reads back all 151,936 logits for the last position and synchronizes.
Loading, startup, tokenization and a separate full-logit audit are excluded.
Weights and activations use the selected compact dtype, with wider arithmetic
where required by each implementation. Both engines use ten CPU threads;
PyTorch uses one inter-op thread and eager Transformers SDPA, without compilation
or quantization. DeepJ uses a 6 GiB heap maximum. `VECLIB_MAXIMUM_THREADS=10`
and `PYTORCH_ENABLE_MPS_FALLBACK=0` are set.

The checkpoint revision is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, containing
1,720,574,976 parameters. [inputs.json](inputs.json) contains the token IDs.
macOS photo analysis and Spotlight jobs were temporarily paused with permission
and resumed after each comparison. Accepted runs recorded no new swap-out.
The earlier samples affected by background GPU/CPU activity are excluded.

## Implementation

Eligible CPU FP16 matrix-matrix products pass compact operands directly to
Accelerate BNNS. Each existing CPU tile uses one native worker and at most 4 MiB
of workspace. Packed row/column storage, offsets and output strides are supported.
Products with fewer than 16 rows or less than 1,048,576 multiply terms retain
the previous routes, avoiding setup overhead in small grouped-attention products.
BF16, mixed and unsupported products retain their existing routes. A cancellation
control checks accumulation before final FP16 rounding; gradients remain FLOAT32.

Eligible zero-offset Metal compact products import their existing `MTLBuffer`
directly into MPSGraph. Nonzero offsets retain MPSMatrix imports. Full captured
Qwen logits were bit-identical between those two import mechanisms.
Compact MPSGraph products use the GPU compilation setting also used by
[PyTorch 2.8.0's MPS stream](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/mps/MPSStream.mm).

Every eight compact graph products, Apple's public
[`commitAndContinue`](https://developer.apple.com/documentation/metalperformanceshaders/mpscommandbuffer/commitandcontinue())
submits an ordered prefix while the host encodes later operations. No intermediate
CPU wait is added. Every submitted buffer is retained, completed and checked
before returning; failure cleanup also drains submitted work. This scheduling
preserves operation order and avoids duplicate model storage. Pure FP32 products
retain their existing execution route.

Shared cache-prefix products use capacity-strided tensor views instead of gathering
K/V copies. A separate inference-only Metal capability combines score products,
scaling, softmax and value products in one dispatch for FP16/BF16 head widths up
to 128 and cache lengths up to 512, including grouped queries. It preserves the
intermediate compact rounding policy and uses hardware half conversions for
normal values with explicit subnormal handling. Other shapes use existing cache
operations. Gradient-enabled execution does not select this inference kernel.

[PyTorch's MPS matmul](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/LinearAlgebra.mm)
and [CPU compact kernels](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/BlasKernel.cpp)
were also reviewed. Repeated buffer-wrapper caching and alternate vector routes
were tested and excluded after they failed to show a benefit. Python is reference
and benchmark tooling, not part of DeepJ inference.

## Precision and regression checks

The full checkpoint audit captures all logits after the prompt and every fixed
continuation token. The table shows the CPU pair and the final reversed-order
Metal pair; both Metal audits are retained in [results.json](results.json).
Compact results are not bit-identical to PyTorch. Fixed continuation tokens keep
the workload the same even when a small logit difference changes the argmax.

| Backend | Dtype | Maximum absolute logit error | Mean absolute logit error | Matching argmax positions |
| --- | --- | ---: | ---: | ---: |
| CPU | FP16 | 0.162598 | 0.00788231 | 16/17 |
| CPU | BF16 | 0.8125 | 0.0541158 | 16/17 |
| Metal / MPS | FP16 | 0.109375 | 0.00674613 | 17/17 |
| Metal / MPS | BF16 | 0.5 | 0.0453323 | 16/17 |

Separate FP32 Qwen, Loop and Prism controls on CPU and Metal retained identical
complete loss sequences and final parameter/gradient hashes against `5e9156d6`.
These are numerical regression controls, not a new training speed comparison.
The CPU timing artifact and final Metal artifact differ only in their Metal
native library; CPU execution bytes and all other JAR entries are identical.

The full hardware-backed suite completed **2,925 tests**, with
zero failures/errors and 26 optional skips. Java line coverage was
94.22%, native line coverage 97.06%, native branch coverage 87.09%,
and native function coverage 100.00%. Metal kernel selection covered
84 of 84 kernels. Java-only CPU fallback checks also passed.
Checkstyle, PMD, duplication, function-size, complexity, frontend and native
readability gates passed.

Tests cover cache strides, grouped heads, unused poisoned slots, leading views,
all compact representable input values, long dependency chains, FLOAT32 gradients,
reused buffer contents and cleanup after an invalid command stream.

## Image, video and Gemma precision

| Model or pipeline | Text encoder | Main model | Sampler and image/frame decoder |
| --- | --- | --- | --- |
| Gemma 2 chat | — | Selected FP32, FP16 or BF16; BF16 default | — |
| SANA-Sprint | BF16 Gemma | Selected FP32, FP16 or BF16 denoiser | FP32 |
| SANA-Video | Selected FP32, FP16 or BF16 Gemma | Selected FP32, FP16 or BF16 denoiser | FP32 |

Gemma already uses compact activations and shares these optimized backend paths.
Image/video CLIs now accept `--dtype float16`, `--dtype bfloat16` and
`--dtype float32`; the desktop settings expose the same choices using the existing
styled selects. Video requests carry precision into the separate worker process.
Image/video defaults and original `Options` constructors remain FP32. New
constructors accept a final `DType`. Cached/streamed working weights follow the
requested dtype. Checkpoint files are unchanged.

Independent fixtures from PyTorch 2.8.0, Transformers 5.22.1 and Diffusers 0.42.1
check compact Gemma features and SANA image/video denoiser predictions on CPU
and Metal, including streamed video weights. Their absolute tolerances are 0.006
for FP16 and 0.04 for BF16. Tiny two-step pipelines exercise FP32 sampling and
image/frame decoding. Integration tests cover the real JavaFX image bridge,
all three video-worker dtypes and staged versus complete video attention.

These checks do not measure full pretrained compact image/video quality or speed.
The earlier full-checkpoint FP32 media comparisons retain their precision scope.
No long pretrained video generation was run for this follow-up.

[results.json](results.json) records every accepted trial, process metrics,
artifact/source fingerprints, numerical controls and excluded exploratory samples.
The [evidence archive](evidence.tar.gz) contains probes, references, launch scripts
and verification output.
