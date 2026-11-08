# CPU photo and video generation

DeepJ completes the small pretrained CPU workloads below faster than the measured
PyTorch references in FP32, FP16 and BF16. The changes remove repeated layout work
and sequential compact pointwise loops. They preserve the existing scalar
arithmetic, model implementations and Metal execution.

## Configuration and timing

Apple M4, 16 GiB unified memory, macOS 15.5, Java 20, ten CPU threads per engine,
PyTorch 2.8.0, Transformers 5.22.1 and Diffusers 0.42.1. The source baseline is
`8eb703abe7570f8dd40d28e6a09da1b9d228afc8`.

These probes use full pretrained weights with deliberately small spatial inputs:

- SANA-Sprint: 64×64, two steps, guidance 4.5, seed 42, the saved fox prompt.
- SANA-Video: 32×32, five frames, one step, guidance 6, seed 42, the saved moving-fox prompt.
- Both engines receive identical prompt IDs, masks and Java-generated noise.
- Photo Gemma encoding remains BF16; the selected precision applies to the
  denoiser and DC-AE. Photo sampler state remains FP32.
- Video Gemma and denoising use the selected precision. Wan decoding and solver
  state remain FP32. FP16 uses the existing range-preserving FP32 Q/K projection,
  attention-denominator and gated-residual policy in both engines.

The timer includes staged weight loading, prompt encoding, sampling and RGB
decoding. It excludes imports, backend/executor initialization and PNG export.
Processes are fresh, but the filesystem cache is not deliberately flushed.
Java uses `-Xms256m -Xmx8g`; IntelliJ, Chrome and the desktop model UI are closed
for the media timings. Each process is bounded to 60 seconds, 9 GiB RSS and
32 MiB additional swap-out. Accepted generation and isolated-stage runs caused
no additional swap.

| Workload | Precision | Final DeepJ confirmation | Python reference |
|---|---|---:|---:|
| SANA-Sprint image | FP32 | **19.77 s** | 32.80 s |
| SANA-Sprint image | FP16 | **19.58 s** | 31.61 s |
| SANA-Sprint image | BF16 | **19.44 s** | 31.71 s |
| SANA-Video preview | FP32 | **19.57 s** | 35.56 s |
| SANA-Video preview | FP16 | **27.09 s** | 62.42 s¹ |
| SANA-Video preview | BF16 | **35.11 s** | 60.62 s¹ |

The DeepJ column uses one fresh process per case with the final code. Earlier
optimized observations are retained separately in [results.json](results.json).
Python photo and FP32 video entries are medians of two complete observations.
These are short workload measurements, rather than statistical significance
claims or measurements of full-resolution CPU generation.

¹ The monolithic compact Python video processes exceeded the 60-second process
guard. Each of their three stages was then run separately with exact FP32
serialization of the selected-precision features and solver latent. These entries
are sums of measured stages, excluding handoff serialization; they are not
completed monolithic timings. The separate runs also provide the complete frames
for accuracy comparisons. The reference uses official Transformers/Diffusers
models with streamed weight assignment to fit memory, and the matched FP16 policy
described above.

Earlier undersized-heap and memory-pressure pilots, diagnostic profiles and
timed-out processes are excluded from the table and identified in the raw data.
IntelliJ was reopened after media timing; the user's training was not resumed.

## CPU implementation

Contiguous compact unary operations previously used one scalar loop and computed
logical tensor addresses for every element. They now use the existing CPU range
scheduler and direct storage offsets. Contiguous dtype conversions reuse that
loop. The existing strided binary row traversal now supports typed storage,
including mixed-precision broadcasts, rather than falling back to per-element
logical-coordinate division. Host allocations are initialized before workers run.

No native kernel, model, autograd, optimizer, checkpoint format or Metal file
changes. A SHA-256 audit preserves all 155 recorded model/Metal files. The
operation functions and per-value rounding remain unchanged; no reduction is
parallelized or reassociated by this patch.

A `[1,596,9216]` GELU diagnostic, with three warm-ups and seven timed observations,
measured medians of 226.71→33.06 ms in FP16 and 118.83→14.36 ms in BF16. The existing
FP32 route stayed at 11.61→11.62 ms. These operator timings explain part of the
pipeline improvement; they are not complete-generation measurements.

## Output regression and Python accuracy controls

All three final CPU photos are pixel-identical to the pre-optimization DeepJ
photos. Replaying identical Python prompt features through the old and new video
sampler/decoder produces identical latent bytes and all five identical frames in
every precision. These checks isolate the CPU optimization from cross-engine
rounding differences.

Full pipelines, with each engine computing its own prompt features, differ by
the following absolute RGB channel levels on the 0–255 scale:

| Workload | Precision | Mean | Maximum | 99th percentile |
|---|---|---:|---:|---:|
| Photo | FP32 | 21.16292 | 254 | 157 |
| Photo | FP16 | 21.02157 | 234 | 140.13 |
| Photo | BF16 | 15.68660 | 235 | 119 |
| Video | FP32 | 0.00020 | 1 | 0 |
| Video | FP16 | 0.12806 | 1 | 1 |
| Video | BF16 | 2.07891 | 15 | 9 |

The deliberately tiny photo latent is only 2×2. Controls trace its sensitivity to
BF16 prompt features and compact arithmetic, rather than a change in the sampler:

- Identical Python prompt features reduce the FP32 DeepJ/Python photo difference
  to mean 0.00716 and maximum 1. The Python replay reproduces its complete original
  photo exactly, validating the feature/noise handoff.
- Python CPU versus saved Python MPS BF16 prompt features differ by mean 0.01843
  over valid features; DeepJ CPU versus Python CPU differs by 0.01734. The saved MPS
  control uses identical prompt IDs, masks and software versions.
- Feeding those two Python feature sets through the same CPU photo implementation
  changes FP32 photo pixels by mean 17.85742 and maximum 238. Both photo pipelines
  use BF16 prompt encoding even when their denoiser precision is FP32.
- Separating Python's linear and convolution bias additions, changing only the
  intermediate rounding points, changes the same-feature FP16 photo by mean
  22.12199 and BF16 by 18.55550. DeepJ versus unmodified Python with the same features
  differs by 13.09774 and 15.69873, respectively. Their final latent relative RMS
  errors are 0.12013 and 0.18806; the Python-only rounding controls give 0.22491 and
  0.21154. These controls demonstrate amplification of compact rounding in this
  minimal spatial example; they do not describe the full-resolution photo results.
- For BF16 video with identical prompt features, DeepJ versus Python averages
  1.70254 levels. Python's separate-linear-bias control alone averages 1.63841;
  matching that composition reduces DeepJ's difference to 1.36868. Independent
  component, sampler-history and rendered-frame Python oracles also pass in the
  complete test suite.

The full-resolution Metal photo and three-step video results remain in their
existing reports. This CPU patch preserves its previous results exactly; it does
not make compact outputs pixel-identical across engines.

## Training and build regression

Before/after one-step audits of small Qwen and Gemma in all three precisions preserve
identical losses and exact tensor payloads for logits, gradients and updated
weights. FP16/BF16 FP32 optimizer master weights also match exactly. Safetensors
comparisons normalize header ordering and compare each tensor's dtype, shape and
raw payload, rather than comparing JSON header byte order across JVM processes.

Separate FP32 Loop, Prism and Qwen controls run 101 training steps each. Loss
sequences and final parameter/gradient hashes match exactly. Their observed
before/after step medians are 3.023/2.971, 1.787/1.784 and 1.614/1.591 ms,
respectively; these short regression controls are not new PyTorch comparisons.

`mvn clean verify -Pfull-coverage -Dsanavideo.test.metal=true` passed through
IntelliJ's terminal: 3,151 tests, zero failures/errors and 28 optional skips.
Java/native coverage gates, all Metal kernel selections, Checkstyle, PMD,
copy-paste detection and frontend quality checks passed. Full Java coverage is
94.29% lines / 83.94% branches; native coverage is 97.16% lines / 87.60% branches.

The added tests cover every storage pair, offset/strided/overlapping views, empty
inputs, worker boundaries, deferred host allocations and unchanged gradients.
[Raw measurements and comparisons](results.json), [reference requirements](requirements.txt)
and [reproduction evidence](evidence.tar.gz) retain the inputs, probes, output
frames, stage logs and guards. Python is reference tooling, not a DeepJ runtime
dependency.

For example, unpack the evidence, compile `harness/CpuMediaProbe.java` against
the built DeepJ classes, and use the same downloaded checkpoint for both engines:

```sh
javac -cp target/classes -d PROBE_CLASSES EVIDENCE/harness/CpuMediaProbe.java
java -Xmx8g --enable-native-access=ALL-UNNAMED \
  -cp target/classes:PROBE_CLASSES CpuMediaProbe image MODEL DEEPJ_OUTPUT FLOAT32
python EVIDENCE/harness/image_reference.py MODEL EVIDENCE/inputs PYTHON_OUTPUT FLOAT32
```

Video references expect `video-input.json` and `noise.f32` in their output folder.
Run `video_text_reference.py`, `video_diffusion_reference.py` and
`video_decode_reference.py` in that order to reproduce the compact stage totals,
passing `MODEL OUTPUT FLOAT16 --safe-fp16` or `MODEL OUTPUT BFLOAT16 --safe-fp16`
to each. The same staged reference is also available as `video_reference.py`.
