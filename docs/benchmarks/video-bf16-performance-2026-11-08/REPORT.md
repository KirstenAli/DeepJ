# Bounded SANA-Video inference on Apple M4

This follow-up closes the BF16 speed gap observed in the earlier
[three-step correctness audit](../video-lifetime-2026-11-08/REPORT.md). In this
matched small preview, DeepJ completed generation sooner than the staged Python
reference in FP32, FP16 and BF16. It does not measure full-size video generation.

## Complete generation

Apple M4, 16 GiB unified memory, macOS 15.5, Java 20, PyTorch 2.8.0,
Transformers 5.22.1 and Diffusers 0.42.1. Both engines use the same checkpoint,
saved prompt IDs and masks, seed-42 starting noise, 192×128 output, five frames,
three solver steps and guidance 6. Captions and guided predictions use batch two.
The prompt is:

> A small red fox walks slowly through a sunny green meadow. Clear natural daylight, detailed fur, steady camera. motion score: 30.

| Precision | DeepJ Metal | PyTorch / Diffusers MPS | Fresh processes per engine |
| --- | ---: | ---: | ---: |
| FP32 | **22.20 s** | 35.89 s | 1 |
| FP16 | **16.93 s** | 24.84 s | 1 |
| BF16 | **15.65 s** | 17.51 s | 2 |

BF16 values are medians of two fresh processes per engine, ordered DeepJ,
Python, Python, DeepJ. Individual times were 15.91098 and 15.39788 seconds for
DeepJ, and 17.64330 and 17.37887 for Python. FP32 and FP16 are single controls;
they are observations rather than repeated performance estimates. FP32 ran
DeepJ then Python; FP16 ran Python then DeepJ.

Timing includes model-stage loading, Gemma prompt encoding, all three guided
denoising steps and Wan decoding through RGB output. It excludes runtime/backend
initialization, Python imports and PNG export. Each engine releases one model
stage before loading the next. Python streams Gemma in all precisions and the
FP32 denoiser; compact denoisers remain resident. These are measurements of the
documented staged reference, including its loading policy, rather than a claim
about every possible Diffusers loading or compilation configuration.

The selected dtype applies to Gemma and the denoiser. Both engines retain FP32
sampler state and Wan decoding. The Python scheduler uses CPU FP32 arithmetic
and copied MPS timesteps to avoid the scalar-view defect reproduced in the
earlier audit. FP16 has explicit reference adjustments matching DeepJ's existing
FP32 Q/K projections, normalization, attention denominator and gated residual
accumulation. Its timing therefore describes that stability policy, not
unmodified upstream FP16 execution. Rotary buffers retain constructor precision.

IntelliJ and Chrome were closed. All eight accepted generation processes finished
without additional swap; guards stopped runs at 60 seconds, 7.5 GiB sampled RSS
or 32 MiB additional swap-out. DeepJ used `-Xmx2g`, ten CPU threads, the unchanged
6 GiB managed Metal limit and its automatic 3 GiB weight-cache budget. RSS is not
a complete measure of shared GPU memory. [Raw results and guards](results.json)
retain stage times, individual trials and output comparisons.

## Implementation

- Repeated video blocks keep complete, owned working-weight groups that fit the
  existing cache budget. Remaining groups stream. Borrowed leases preserve the
  cache's ownership when an individual reader closes. Single-pass Gemma keeps
  its existing streaming selection. This uses more of the already allowed cache
  capacity than the former all-or-nothing policy; it does not raise the budget.
- Eligible contiguous BF16 device projections flatten independent batch rows
  into one matrix before calling the existing native compact product. This
  avoids the previous generic product's FP32 staging. CPU, other precisions,
  strided/offset views and gradient-enabled calculations retain their old route.
  PyTorch's [`_flatten_nd_linear`](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Linear.cpp)
  similarly groups contiguous leading dimensions for linear projection.
- Small loaded video blocks capture bounded GPU work together rather than
  cleaning temporary resources between each stage. Blocks with more than 512
  combined batch/token rows or 1,048,576 input elements retain staged cleanup.
  Loaded blocks already require `noGrad`; the differentiable supplied-weight
  path is unchanged. Retained blocks close before Wan decoding begins.

No native kernel, optimizer, tensor/autograd implementation, Loop, Prism or
Qwen calculation changes are included. Gemma's eligible no-gradient BF16 batch
projections use the shared projection helper; its training branch is unchanged.

## Output and regression controls

Every engine computes its own prompt features, denoiser predictions, solver
trajectory and rendered frames. Absolute RGB channel differences on the 0–255
scale across all five output frames were:

| Precision | Mean DeepJ versus Python | Maximum | 99th percentile |
| --- | ---: | ---: | ---: |
| FP32 | 0.000377 | 1 | 0 |
| FP16 | 0.510943 | 12 | 4 |
| BF16 | 5.584660 | 83 | 34 |

Both BF16 engines repeated their own frames exactly. The previous BF16 audit
averaged 5.568812 levels against Python, with maximum 78. Its Python-only change
to linear/bias rounding averaged 4.81024 levels and reached 71. The current
BF16 comparison is of similar scale; the faster execution does not remove
the already measured compact-rounding sensitivity. The current versus previous
DeepJ outputs average 0.001012 / 0.199655 / 2.102420 levels in FP32 / FP16 / BF16,
with maxima 1 / 5 / 41. Execution grouping can change vendor accumulation and
intermediate rounding, despite preserving the formulas and selected dtypes.

Two controls isolate the new execution routes:

- A real transposed projection, `[2, 192, 2240] × [2240, 2240]`, matches the
  equivalent PyTorch BF16 matrix result in all 860,160 elements. The representative
  local product median fell from 3.847 to 3.585 ms including full output readback.
  This component measurement is separate from complete generation.
- A pretrained block at the actual preview width compares staged and grouped
  execution with fixed identical inputs. Relative RMS differences are
  0.000000997 / 0.000516650 / 0.004265846 in FP32 / FP16 / BF16. The block's
  unscaled outputs are much larger than RGB values; these are relative tensor
  errors, not colour-level differences. [Component results](components.json).

The cache regression closes borrowed readers, deletes their checkpoint, cleans
temporary buffers and verifies that admitted weights survive while uncached
blocks still require the checkpoint. Repeated cached and streamed predictions
match exactly in each dtype. Projection tests check eligible BF16 shapes,
noncontiguous and offset views, other precisions, and identical forward values
and input/weight gradients with gradient recording enabled.

Forty focused tests passed on physical Metal, including the independent compact
video oracle, solver-history retention, photo sampler regression and sequential
guidance controls. Full verification reported 3,126 tests, zero failures/errors
and 29 optional skips, with Java/native coverage and all style checks passing.
The validation uses small examples; no long or 81-frame generation was run.

## Reproduction

[evidence.tar.gz](evidence.tar.gz) contains the exact benchmark probes, staged
Python reference and helpers, inputs, guard records, logs and accepted frames.
They are test/reference tooling; Python is not part of DeepJ inference.
The source baseline is `ab8841a1893bd175f48cbc185aaf03bd9975d52c` plus the code
changes accompanying this report. [requirements.txt](requirements.txt) pins the
Python environment. After unpacking the evidence, compile `VideoAuditProbe.java`
against the built DeepJ classes and use the same model directory for both engines.
Copy `inputs/video-input.json` and `inputs/noise.f32` into each output directory.

```sh
java -Xmx2g --enable-native-access=ALL-UNNAMED \
  -Ddeepj.metal.maxBufferBytes=6442450944 \
  -cp target/classes:PROBE_CLASSES VideoAuditProbe video MODEL OUTPUT BFLOAT16
python video_reference_cpu_scheduler.py MODEL OUTPUT BFLOAT16 --safe-fp16

mvn test -Dsanavideo.test.metal=true \
  -Dtest=VideoBlockCacheTest,VideoWeightCacheTest,CompactVideoReferenceTest,ModelProjectionTest,VideoGuidanceTest,SanaVideoTrajectoryTest,CompactMediaPipelineTest,SanaVideoEndToEndReferenceTest
mvn clean verify -Pfull-coverage
```
