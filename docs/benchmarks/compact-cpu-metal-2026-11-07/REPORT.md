# Shared CPU and Metal compact execution

Measured on 7 November 2026 on an Apple M4 with 16 GiB unified memory,
macOS 15.5, Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
These changes optimize shared tensor operations. Qwen, Loop and Prism use the
same backend operations and retain FLOAT32 gradients and optimizer state.

## Full Qwen3-1.7B inference

Both engines use the same checkpoint, parameter dtype, fixed token IDs and KV
cache. The model's activations remain in the selected compact dtype, with wider
arithmetic where required by each implementation. These are not weight-only
16-bit measurements.

| Backend | Dtype | DeepJ prompt (ms) | PyTorch prompt (ms) | DeepJ decode (tokens/s) | PyTorch decode (tokens/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | FLOAT16 | 945.7 | **719.0** | **22.06** | 20.73 |
| CPU | BFLOAT16 | **918.9** | 1050.5 | **22.23** | 19.85 |
| Metal / MPS | FLOAT16 | 109.2 | **108.0** | 20.58 | **23.16** |
| Metal / MPS | BFLOAT16 | 109.6 | **107.7** | 21.22 | **23.06** |

Each value is the median of five measured trials after five warm-up trials in
one fresh process. CPU decoding is faster in both compact formats in this test.
CPU FLOAT16 prompt processing and Metal compact decoding still trail PyTorch;
these results do not establish universal parity or a general speedup ratio.
Metal prompt times are close in these samples, without a statistical equivalence claim.

The batch size is one, the prompt has 64 tokens, and decoding processes the same
16 fixed continuation tokens without sampling. Each forward reads back all logits
for the last position, synchronizing device work. Prompt processing and decoding
are timed separately. Loading, startup, tokenization and the separate logit audit
are excluded. PyTorch uses eager Transformers SDPA, without compilation or
quantization. Both engines use ten CPU threads; PyTorch uses one inter-op thread.
`VECLIB_MAXIMUM_THREADS=10` is set. DeepJ's compact runs use a 6 GiB heap maximum.

The full model is Qwen3-1.7B checkpoint revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with 1,720,574,976 parameters.
[inputs.json](inputs.json) contains the token IDs. [results.json](results.json)
contains every trial, checksums, artifact fingerprints and process memory records.

## Implementation

- CPU compact matrix-vector multiplication reads compact weights directly with
  SIMD and FLOAT32 accumulation. Matrix-matrix multiplication uses bounded
  converted tiles and Accelerate BLAS. Scratch output is initialized only where
  an accumulating vector kernel needs it.
- Compact addition, SwiGLU forward/backward, normalization and softmax use native
  CPU loops. Normalization shares scalar FLOAT32 statistics with the existing
  path; numerically sensitive exponential rows retain Java fallback.
- Compact rotary operations access two-byte arrays directly. Contiguous inner
  blocks in tensor copies use array copies while retaining view and broadcast semantics.
- Eligible Metal FLOAT16/BFLOAT16 matrix products use MPSGraph directly. The route
  requires macOS 14+, Apple-family-7 hardware or newer, aligned packed storage,
  inner/output dimensions at least 256, and either one row or at least 64 rows.
  Unsupported shapes retain existing MPS or custom-kernel routes.
- Metal compact normalization uses parallel row reductions. Eligible compact
  checkpoint projections use column-major storage; FLOAT32 loading is unchanged.
- `ParameterFactory.uniform(dtype)` enables compact parameters and embedding
  activations for models accepting a factory. Explicit compact Qwen CLI/UI dtype
  selections use the shared activation policy. Existing API overloads, omitted
  CLI dtype and the desktop FLOAT32 default keep their prior policy.

The activation policy is a construction choice, not a new checkpoint field.
Restore weights into a model constructed with the same policy; see
[parameter and activation precision](../../tensors.md).

A larger CPU staging allocation and a larger Metal buffer pool were tested and
excluded because they did not show a consistent benefit. CPU compact matrix-matrix
work still widens tiles for BLAS; this is not a claim of native half-precision
arithmetic throughout the CPU forward pass.

## Precision checks

The full checkpoint audit compares all 151,936 logits after the prompt and after
each of the 16 fixed tokens. Differences are recorded separately from timing.
Compact outputs are not bit-identical to PyTorch; small logit differences can
change selected tokens. The audit uses fixed tokens so both engines still perform
the same workload. Tiny checkpoint references also test both compact dtypes,
cached versus uncached execution, checkpoint loading, streaming and training.

| Backend | Dtype | Maximum absolute error | Mean absolute error | Matching argmax positions |
| --- | --- | ---: | ---: | ---: |
| CPU | FLOAT16 | 0.101563 | 0.00651916 | 16/17 |
| CPU | BFLOAT16 | 0.8125 | 0.0541158 | 16/17 |
| Metal | FLOAT16 | 0.109375 | 0.00674613 | 17/17 |
| Metal | BFLOAT16 | 0.5 | 0.0453323 | 16/17 |

## FP32 training and regression controls

The [matched small-Qwen workload](../qwen-training-2026-11-07/REPORT.md#matched-workload-and-timing)
uses 1,640,192 parameters, width 256, two blocks, batch size one and 64-token
sequences. Timings include forward, cross-entropy, backward, gradient clipping,
AdamW, loss readback and synchronization. Each fresh process uses 100 warm-up
updates and 30 timed updates. Both engines load the same initial parameters and
inputs. Each summary is the median of two process medians, with the second pair
reversing engine order. This measures FP32 training, not full Qwen3-1.7B training
or a compact-training speed comparison.

| Backend | DeepJ (ms/step) | PyTorch (ms/step) |
| --- | ---: | ---: |
| CPU | **6.718** | 9.016 |
| Metal / MPS | **4.113** | 17.824 |

| Engine | First process median | Second process median |
| --- | ---: | ---: |
| DeepJ CPU | 6.686 ms | 6.750 ms |
| PyTorch CPU | 8.947 ms | 9.085 ms |
| DeepJ Metal | 4.132 ms | 4.093 ms |
| PyTorch MPS | 17.688 ms | 17.960 ms |

DeepJ's CPU and Metal step times are lower in both matched pairs. The earlier
[numerical comparison and gradient-clipping reduction differences](../qwen-training-2026-11-07/REPORT.md#numerical-checks)
still apply; these measurements do not establish identical long-run training.

Separate before/after FP32 controls compare the final release JAR against
`4ed97933`, using 100 warm-up updates and 30 timed updates. All six controls retain
identical loss sequences and final parameter/gradient SHA-256 hashes.

| Backend | Model | Before (ms/step) | After (ms/step) |
| --- | --- | ---: | ---: |
| CPU | Qwen | 11.607 | 6.709 |
| CPU | Loop | 10.291 | 5.933 |
| CPU | Prism | 9.747 | 5.784 |
| Metal | Qwen | 4.114 | 4.060 |
| Metal | Loop | 4.303 | 4.289 |
| Metal | Prism | 4.540 | 4.593 |

The small Metal timing movements in these controls do not establish a regression
or a speedup. The loss and hash checks establish preserved results for these
bounded FP32 workloads; they are not a guarantee for every training configuration.

Full verification on physical Metal passed 2,884 tests, with zero failures/errors
and 29 optional skips. Checkstyle, PMD, duplication, complexity and coverage gates
passed. Java line coverage was 94.16%, native line coverage 97.15%, and Metal kernel
selection coverage 100%. All 32 targeted Java-only fallback checks passed.
Compact Qwen, Loop and Prism training tests passed on CPU and Metal in both dtypes.

## Resource handling and provenance

Full-model runs execute sequentially with IntelliJ, Chrome and Zoom closed.
A guard stops an engine at 180 seconds, 11 GiB RSS or 128 MiB new system swap-out.
All eight accepted compact runs recorded zero new swap-out. RSS does not include
all driver allocations and is not used to claim a memory advantage.

A fresh DeepJ CPU FLOAT32 control measured 290.6 ms prompt processing and
12.68 tokens/s. The following PyTorch FLOAT32 process crossed the swap guard while
loading and was stopped before timing. It is excluded from comparisons. The
README's full FLOAT32 inference comparison remains the earlier
[complete matched run](../full-qwen-2026-11-07/REPORT.md).

The compact inference samples used the integrated JAR plus a Metal loading-layout
class override, whose implementation was subsequently included in `MetalCoreOps`.
CPU classes came directly from that JAR. The final source additionally restricts
MPSGraph for small matrices to preserve their established accumulation order;
this does not change the 64-row prompt or one-row decode route in this benchmark.
The final artifact is checked separately by the regression suite and training
controls. The measured and final artifact hashes are recorded rather than
attributing old measurements to a newer build.

[The evidence archive](evidence.tar.gz) contains the actual probe sources, launch scripts and logs.
Replace local paths when replaying them; build native libraries before packaging
so the JAR contains the rebuilt binaries. The repository's `Qwen3Benchmark` also
accepts a final `true` argument for compact activations, following thread count,
warm-up count and measured-run count. Python code is benchmark/reference tooling,
not part of DeepJ's runtime.
