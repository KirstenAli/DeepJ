# Compact inference and chat on Apple M4

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5, Temurin 20.0.2,
PyTorch 2.8.0 and Transformers 4.51.3. This follows `a8625200` on
`feature/shared-tensor-performance`. The changes improve shared tensor operations
and compact Metal execution. Python is benchmark/reference tooling, outside the Java runtime.

## Qwen3-1.7B decode throughput

| Precision | DeepJ Metal (tokens/s) | PyTorch MPS (tokens/s) | DeepJ prompt (ms) | PyTorch prompt (ms) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | **24.00** | 23.48 | **104.7** | 106.1 |
| BF16 | **24.02** | 23.63 | **104.2** | 105.6 |

DeepJ had higher decode medians in both formats in both engine-order pairs.
The combined point estimates are 2.2% higher for FP16 and 1.7% higher for BF16.
These are short workload measurements, without a claim of statistical significance
or a universal speed advantage across models, prompts, hardware or library versions.

Each fresh process runs five warm-up trials and five timed trials. Two fresh
processes per engine and dtype are summarized by the median of their process
medians; the second pair reverses engine order. Both engines use the same checkpoint
and selected weight/activation dtype, batch size one, a KV cache, a 64-token prompt
and the same 16 fixed continuation token IDs. There is no sampling. Each step
reads back all 151,936 logits and synchronizes. Loading, tokenization and a separate
full-logit audit are outside the timing. Wider arithmetic is retained where required
by each engine. Both use ten CPU threads; PyTorch uses one inter-op thread and eager
Transformers SDPA, without compilation or quantization. DeepJ uses `-Xmx6g`.
`VECLIB_MAXIMUM_THREADS=10` and `PYTORCH_ENABLE_MPS_FALLBACK=0` are set.

The checkpoint revision is `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with
1,720,574,976 parameters. Exact inputs are in [inputs.json](inputs.json). macOS photo
analysis and Spotlight were paused with permission and resumed after every run.
Accepted runs recorded no new swap-out.

| Precision | Pair | DeepJ decode (tokens/s) | PyTorch decode (tokens/s) |
| --- | ---: | ---: | ---: |
| FP16 | 1 | 23.90 | 23.57 |
| FP16 | 2 | 24.09 | 23.40 |
| BF16 | 1 | 23.95 | 23.61 |
| BF16 | 2 | 24.09 | 23.65 |

## Streaming chat

This separate check exercises the normal Qwen chat adapter and Transformers
`generate`, including token selection and cumulative text decoding. Both receive
identical 33-token non-thinking chat input IDs, then generate 160 greedy tokens.
A short 16-token warm-up and a full 160-token warm-up precede two timed requests.
The table summarizes those two requests. Loading is excluded.

| Precision | DeepJ (tokens/s) | PyTorch (tokens/s) | DeepJ first token (ms) | PyTorch first token (ms) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | **22.93** | 22.40 | 94.0 | 90.1 |
| BF16 | **23.08** | 22.61 | 90.4 | 90.8 |

Chat throughput is `(tokens - 1) / (last token time - first token time)`. PyTorch
selects the greedy token on-device; DeepJ reads the logit row and selects on CPU.
Both stream and decode their generated text. Naturally generated BF16 answers can
differ; the fixed-continuation test above holds every model input constant.

Before the chat fixes, the same short prompt took about 215 ms to its first token
in DeepJ, and cache expansion produced observed token gaps of 225–445 ms.
The final timed requests had first-token medians of 90–94 ms; their individual
maximum token gaps were below 65 ms. The earlier probes used a shorter warm-up
protocol and are diagnostic observations, rather than a controlled speedup ratio.

## Shared implementation

- Contiguous host snapshots copy typed storage directly, including nonzero offsets.
  Strided snapshots retain logical indexing. The same helper serves full arrays
  and row reads. Compact 151,936-element copies measured about 0.07–0.09 ms in the
  small probe, versus 0.37–0.47 ms previously; these are host-copy measurements.
- Eligible compact Metal products with 2–63 rows now use MPSGraph as well as
  larger products and single-row decoding. Unsupported layouts retain their routes.
  Compact matmul reduction order can differ by a rounding unit from the strided
  kernel; mixed/FLOAT32 routing is preserved. Short products have independent
  output, FP32 accumulation and gradient checks.
- Normal FP16 conversion uses hardware instructions; integer handling retains
  subnormal, infinity and NaN behavior. Tests check every 16-bit value, randomized
  FP32 patterns, all positive/negative compact midpoints and neighboring values.
- Compact normalization preserves its reduction tree while replacing the last
  shared-memory reduction stages with SIMD shuffles. An optional shared backend
  capability returns only normalized output during compact `noGrad` inference,
  avoiding backward-only tensors and allocations. Gradient-enabled execution still
  saves FLOAT32 normalized values and statistics. Automatic FLOAT32 routing is unchanged.
- `InferenceContext.grown(...)` copies existing K/V storage into a larger cache.
  Qwen sessions use it when capacity grows, avoiding prefix recomputation at
  128/256-token boundaries. Growth preserves batches, grouped heads, storage dtype
  and absolute positions; wrapped caches reject growth. Training does not use these caches.

Cached MPSGraph executables and two custom packed-vector prototypes were tested
and excluded after they failed to improve the selected route. Their source and
diagnostic results are retained in the evidence archive; they are outside production code.

## Precision and regression checks

All 17 captured full-vocabulary logit rows remain bit-identical to DeepJ
`a8625200` in both formats. The compact PyTorch comparison remains:

| Precision | Maximum absolute logit error | Mean absolute logit error | Matching argmax positions |
| --- | ---: | ---: | ---: |
| FP16 | 0.109375 | 0.0067461288 | 17/17 |
| BF16 | 0.5 | 0.045332329 | 16/17 |

These compact outputs are not bit-identical to PyTorch. Fixed continuation tokens
preserve the workload even when rounding changes an argmax. Separate CPU and Metal
FP32 Qwen, Loop and Prism training controls retained identical complete loss sequences
and final parameter/gradient hashes against the previous build. The controls are
numerical regression checks, rather than a new training speed comparison.

The final physical-Metal suite completed **2,943 tests**, with zero failures/errors
and 26 optional skips. Java-only CPU fallback checks passed 26 tests. Full Java
line coverage was 94.23%; native line/branch/function coverage was
97.07% / 87.07% / 100.00%, with 84 of 84 Metal kernels selected. Checkstyle, PMD,
duplication, function-size, complexity, frontend and native readability gates passed.
The Clang analysis buffer limit increased to fit the complete source AST; quality
thresholds are unchanged.

## Image and video precision

Sampling keeps FP32 state in both image and video pipelines, matching Diffusers.
The SANA-Sprint DC-AE decoder now follows the selected FP32, FP16 or BF16 precision,
as does the image denoiser. The published [0.6B Python example](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)
uses BF16 weights including DC-AE. The published [SANA-Video example](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/sana_video/pipeline_sana_video.py)
uses a BF16 text encoder/denoiser with an FP32 Wan decoder; DeepJ retains FP32 video
decoding. Existing DeepJ pipeline defaults remain FP32.

Independent compact DC-AE outputs from PyTorch/Diffusers were added to the component
fixtures and checked on CPU and Metal. Tiny compact image/video pipeline checks
and the real JavaFX bridge pass. These checks do not establish full pretrained
compact media quality or speed. No long pretrained video generation was run.

## Evidence

[results.json](results.json) includes every accepted trial, process medians, memory
guards, artifact/source hashes, full-logit error summaries and numerical training
controls. [evidence.tar.gz](evidence.tar.gz) contains probes, references, launchers,
raw timing logs, verification logs and excluded exploratory samples.
The final verified JAR has identical execution classes and native libraries to
the measured JAR; its only changed entry is clarified image-precision helper text.
The earlier CPU compact results remain in the [previous report](../compact-followup-2026-11-07/REPORT.md).
