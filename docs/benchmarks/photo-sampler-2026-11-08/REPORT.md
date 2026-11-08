# SANA-Sprint compact sampler input lifetime

The complete-photo follow-up found and fixed a buffer-lifetime bug in SANA-Sprint
FP16/BF16 generation. The earlier roughly 18.5/255 photo difference was not
explained by the component rounding checks. This report supersedes those compact
photo results; the original measurements remain available as historical data.

## Cause and correction

`SanaSprint.step` computes an FP32 model input, converts it to the selected model
dtype, runs the denoiser, then uses the original FP32 input again in the sampling
formula. Compact conversion creates separate storage. Denoiser cleanup could
release the original FP32 temporary because it was not retained across that call.
The subsequent sampling calculation then lost its input/noise contribution.

At the first sampling angle, near π/2, the expected denoised result is approximately
`0.5 * (noise - prediction)`. The broken compact path instead produced approximately
`-0.5 * prediction`. Saved first-step arrays reproduce this missing term; it is a
sampling error, rather than ordinary compact rounding.

The fix retains the already computed FP32 input until the sampling step finishes.
It does not change the sampling formula, weights, tensor kernels or model math.
At 1024×1024 this retains one existing 128 KiB tensor for the duration of the step.
It introduces no image-model retraining or checkpoint change.

## Isolation controls

All controls use the same local pretrained SANA-Sprint checkpoint and saved fox
prompt IDs, masks and both noise tensors as the earlier comparison. The environment
is Apple M4, 16 GiB unified memory, macOS 15.5, Java 20, PyTorch 2.8.0,
Transformers 5.22.1 and Diffusers 0.42.1.

- Repeating the complete default BF16 Python MPS photo reproduced the saved PNG
  pixel-for-pixel.
- Python's own FP32-to-FP16 photo difference averaged 0.42522 channel levels;
  FP32-to-BF16 averaged 1.31484. These did not justify DeepJ's former 18.5-level gap.
- A complete same-precision Python control separated bias addition from linear
  and convolution products, changing the intermediate rounding point. Its photo
  differences averaged 0.49471 for FP16 and 3.18458 for BF16, with maxima 149 and
  243. This is a deliberate arithmetic-composition variant, not the unmodified
  reference used for the DeepJ timing and quality comparison.
- Feeding identical Python latents to the full 1024×1024 DeepJ decoder produced
  mean channel differences of 0.000019 for FP32, 0.02355 for FP16 and 0.17294 for
  BF16. The decoder was not the source of the large full-photo discrepancy.
- Feeding identical Python prompt features to DeepJ still reproduced the large
  discrepancy before the fix. Retaining the sampler input reduced it to 0.59244
  for FP16 and 2.84748 for BF16, isolating the sampler lifetime error.
- Instrumenting the old two-step sampler did not change its result, ruling out
  diagnostic readback as the explanation. The first compact-to-FP32 prediction
  conversion also preserved every value exactly.

RGB differences use absolute channel differences on the 0–255 scale. Remaining
compact differences are comparable to the complete Python rounding-policy
control; the full images are not claimed to be pixel-identical.

The full CPU FP16 control was stopped at its 90-second time limit before producing
a photo. A whole-pipeline alternate-Metal-kernel Python control was stopped by
the 32 MiB swap-growth guard. Neither supplies a completed quality or timing result.
No long video run was used for this investigation.

## Regression

A seeded 32×32 two-step pipeline compares CPU and Metal outputs in every dtype.
Before the fix, FP16 differed by up to 108 channel levels and BF16 by 113. With
the fix, FP16 matches exactly and BF16 differs by at most one level; FP32 matches
exactly before and after. The permanent regression allows two channel levels to
accommodate small backend rounding differences and fails decisively on the old
code. The earlier pipeline test checked image dimensions and did not catch this.

Only the SANA-Sprint inference pipeline changes. The low-level differentiable
image transformer, video model, Qwen, Gemma, Loop, Prism, optimizers, shared tensor
operations and native kernels are unchanged in this patch.

## Complete photos and timings after the fix

Each engine ran twice in a fresh process for each precision, with the order reversed
in the second pair. Both used the same 1024×1024 fox prompt, two sampling steps,
guidance 4.5, checkpoint values, saved token IDs, masks and noise. Each pipeline
computed its own Gemma features, sampler states, denoiser predictions and decoder
pixels. Image Gemma encoding uses BF16; the selected precision applies to the image
denoiser and DC-AE. Sampler state remains FP32 in both engines.

Timing includes stage loading, prompt encoding, sampling and RGB decoding. It
excludes runtime/backend initialization, Python imports and PNG export. DeepJ
uses a 2 GiB JVM heap and 6 GiB managed Metal allocation cap. Both engines use
ten CPU threads. The numbers are medians of two short fresh-process observations,
not statistical significance claims or predictions for other hardware/prompts.

| Precision | DeepJ Metal | PyTorch/Diffusers MPS | Mean RGB difference | Maximum RGB difference | 99th percentile |
| --- | --- | --- | --- | --- | --- |
| FP32 | 18.04 s | 18.86 s | 0.59943 | 124 | 7 |
| FP16 | 16.70 s | 17.80 s | 1.31804 | 231 | 20 |
| BF16 | 16.39 s | 16.73 s | 2.94526 | 253 | 40 |

Both runs produced identical pixels within each engine and precision. FP32 DeepJ
pixels are also identical to the pre-fix FP32 result. DeepJ's median generation
time was lower in every precision. The compact correction therefore retains the
observed speed advantage in this workload while removing the large sampler error.
There is no pixel-identity claim between engines, and the isolated maxima are
reported alongside averages rather than being omitted.

All twelve accepted generation runs completed without additional swap and within
the process time/RSS guards. One earlier attempt finished generation but failed
PNG export when the disk filled; another hit the swap-growth guard. Both are
recorded separately from accepted trials. Raw timings, stages, guard results,
input hashes and comparison summaries are in [results.json](results.json).

## Build verification

`mvn clean verify -Pfull-coverage` passed from IntelliJ's terminal with physical
Metal and desktop access: 3,105 tests counted, 3,076 passed, 29 optional checkpoint,
diagnostic or model-specific tests skipped. All seven `CompactMediaPipelineTest`
cases executed, including the three new dtype regression cases. Java/native
coverage gates, all Metal kernel selections, Checkstyle, PMD, copy-paste detection
and UI linters passed.
