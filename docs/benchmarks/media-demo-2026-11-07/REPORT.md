# Gemma, image and video demo checks on Apple M4

Follow-up [training, inference and video comparisons](../gemma-video-gap-2026-11-07/REPORT.md)
include the restored original FP32 fox and later loading improvements.

7 November 2026. Apple M4, 16 GiB unified memory, macOS 15.5,
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3. These checks follow
`0eccc7c1` on `feature/shared-tensor-performance`.

## Gemma 2 2B-IT

The complete 2,614,341,888-parameter Gemma checkpoint supplied with SANA-Sprint
was loaded in both engines. Both receive the same 21-token formatted chat prompt
and 16 fixed continuation token IDs. Batch size is one, the KV cache is enabled,
and each step reads all 256,000 logits back to the CPU. There is no sampling.
Model loading and tokenization are outside the timed section. Each engine runs
three warm-up trials followed by three measured trials in a fresh process.

| Precision | DeepJ Metal decode (tokens/s) | PyTorch MPS decode (tokens/s) | DeepJ prompt (ms) | PyTorch prompt (ms) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 15.41 | 15.53 | 107.2 | 96.7 |
| BF16 | 15.04 | 15.37 | 131.7 | 97.3 |

Each cell is the median of the three timed trials. The final pair ran DeepJ
before PyTorch for each dtype. These are short diagnostic measurements with one
fresh process per engine and dtype. They do not establish a statistically
significant speed difference. DeepJ's decode medians are 0.7% and 2.1% lower;
its prompt processing remains slower in both formats.

Both engines use ten CPU threads, with one inter-op thread in PyTorch.
`PYTORCH_ENABLE_MPS_FALLBACK=0` prevents CPU fallback in the reference.
Transformers uses eager attention to retain Gemma's attention score softcap.
DeepJ uses `-Xmx3g`, compact weights and activations, FP32 normalization
statistics, and Gemma's wider softmax calculations. The cache capacity is 128.
Exact inputs, every trial and memory-guard records are in [results.json](results.json).

An initial diagnostic measured about 5.59 tokens/s in DeepJ BF16. The full model
also now produces coherent streamed answers through the normal chat adapter.
The final checks generated 58 BF16 and 64 FP16 visible text updates, with first
text at 144 ms and 112 ms respectively; these single requests are separate from
the fixed-token comparison above.

## Shared fixes

- Aligned tensor views with nonzero offsets exposed an MPSGraph import error:
  the selected row could be replaced by an earlier row. Nonzero offsets now use
  the existing offset-aware matrix kernels. Small single-row left operands can
  first be copied to zero-offset device storage to retain the compact fast path.
  Arbitrary weight views do not require an additional full weight copy.
- Compact `ModelOps.rms(...)` inference reuses the backend's output-only
  normalization kernel. Values, statistics and affine calculations remain FP32
  internally; the result is then converted to the input dtype. This removes the
  composed serial reduction bottleneck. A diagnostic sequence of 105 norms
  dropped from roughly 120 ms to roughly 9 ms. Parallel FP32 summation can change
  compact rounding at a boundary.
- `ModelInference.batch(...)` captures bounded inference work and defers
  per-block materialization. Gemma generation uses it for short inputs;
  prompts longer than 128 tokens retain the existing staged policy. Nested calls
  and failures restore the caller's policy. The scope requires `noGrad`.

The RMS optimization is disabled when gradients are enabled. Tests retain exact
training outputs and every input, gain and bias gradient in all three storage
formats. The shared offset tests also verify view and weight gradients. No CPU
kernel, optimizer, Loop/Prism calculation or FP32 Metal kernel changed here.

## Full-vocabulary numerical checks

Both engines capture the prompt's final row and all 16 continuation rows:
4,352,000 logits per dtype.

| Precision | Mean absolute error | Maximum absolute error | Matching argmax rows |
| --- | ---: | ---: | ---: |
| FP16 | 0.0056843 | 0.046875 | 17/17 |
| BF16 | 0.0335128 | 0.375 | 16/17 |

The remaining BF16 argmax difference is a tie in DeepJ: tokens 2185 and 6011
both score 18.375. PyTorch scores them 18.25 and 18.375. The fixed continuation
keeps subsequent inputs identical regardless of this choice. Compact outputs
are not bit-identical across engines.

Before the offset fix, the prompt row had mean error 3.02 and maximum error
32.375. In the final BF16 run those prompt-row errors are 0.02824 and 0.25,
and the first-token argmax agrees. This large row-selection error is fixed;
it is separate from the compact numerical differences in the table.

## SANA-Sprint fox

Prompt: **A red fox in a snowy forest, soft morning light, detailed wildlife photograph**

Both runs use the original README settings: **1024 × 1024, two steps,
guidance 4.5 and seed 42**, with the complete checkpoint at revision
`aa76e7f4f4928f378716b6716a2130fba3caf5b1`.

| Denoiser and DC-AE precision | Prompt stage (s) | Denoising stages (s) | Image decode (s) | Complete generation (s) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 16.73 | 13.45 | 58.85 | 89.05 |
| BF16 | 16.83 | 12.48 | 51.92 | 81.24 |

These are single fresh-pipeline runs, including stage loading and initialization,
with an additional 0.11 seconds to save each PNG. They are not repeated warm
generation medians or comparisons against Python. Gemma remains BF16 and sampler
state remains FP32 in both runs. DeepJ uses `-Xmx3g`; the managed Metal budget is
8 GiB for FP32 and 6 GiB for BF16. An earlier FP32 attempt with the smaller 6 GiB
test budget correctly stopped at the allocation guard during decoding.

Both latest outputs were visually checked and contain a clear red fox in snow.
The FP32 control before the normalization optimization reproduced the original
README PNG byte-for-byte. The latest normalization path changes floating-point
reduction order, so the latest images are not claimed to be pixel-identical to
the older capture. Output hashes are recorded in `results.json`.

![Latest FP32 fox](fox-fp32.png)

![Latest BF16 fox](fox-bf16.png)

## SANA-Video small execution check

The real pretrained video checkpoint was exercised at **192 × 128, five frames,
one denoising step, guidance 6 and seed 42**. The prompt was:

> A small red fox walks slowly through a sunny green meadow. Clear natural daylight, detailed fur, steady camera. motion score: 30.

| Denoiser precision | Prompt stage (s) | Denoising stage (s) | Frame decode (s) | Complete generation (s) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 29.50 | 24.87 | 8.62 | 63.26 |
| BF16 | 29.41 | 24.75 | 8.68 | 63.09 |

Each run saved five PNG frames and an H.264 MP4 at eight frames per second.
The BF16 row uses the normalization fix; the FP32 row is the earlier control.
These are single fresh-pipeline execution probes, including weight loading.
Gemma and the denoiser use compact weights in the BF16 run, while the sampler
and Wan decoder remain FP32. The FP32 run expands the locally stored BF16
Transformer values to FP32. No long or full-size video was generated in this
check. One denoising step produces an under-generated clip, not demo-quality
fox footage, and these timings do not predict full-clip generation time.

## Verification and evidence

The complete physical-Metal test run passed **2,956 tests** with zero failures
or errors and 57 optional skips. The subsequent focused run passed 43 tests
with zero failures/errors and three optional skips, including the added
offset-view gradient checks. An additional 37 image/video reference and aligned
offset tests passed on Metal with no skips. Checkstyle, PMD, duplication, frontend and native
code-quality gates passed. No quality threshold was weakened.

Accepted model runs recorded no new swap-out. RSS and time guards bounded each
process. RSS is not a complete measurement of GPU or total system memory.
The unsuccessful old-build Gemma comparison was stopped by the swap guard before
usable timings and is excluded. Photo analysis and Spotlight were temporarily
paused with permission and resumed after each batch. Training remained stopped.

[results.json](results.json) contains trials, stages, artifact hashes, logit-error
summaries and guard results. [evidence.tar.gz](evidence.tar.gz) contains the
benchmark probes, references, launchers and verification logs. These are test
tools outside the Java runtime. The separate [Qwen measurements](../demo-compact-2026-11-07/REPORT.md)
use their own model, inputs and timing protocol.
