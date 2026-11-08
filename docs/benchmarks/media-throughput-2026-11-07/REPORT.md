# SANA image and video throughput

The compact photo results in this report predate the
[sampler input-lifetime correction](../photo-sampler-2026-11-08/REPORT.md).
They measured a path that released an FP32 sampler input too early during compact
denoising. Use the corrected full-photo measurements in the follow-up. The
historical trials below are retained; the fix does not change video calculations.

Apple M4, 16 GiB unified memory, macOS 15.5. This follow-up starts from
`aa38704a` on `feature/media-throughput`. It measures complete, staged generation
on Metal/MPS with the official pretrained checkpoints.

## Measured generation time

Lower seconds means faster generation. Photos report the median of two fresh
processes per engine and dtype, reversing engine order in the second pair.
Videos use one fresh process per engine and dtype, Python first, limited to a
five-frame preview. These short runs record lower DeepJ times in each row;
they do not establish performance for every prompt, size or hardware setup.

| Workload | Precision | DeepJ Metal | PyTorch / Diffusers MPS |
|---|---|---:|---:|
| SANA-Sprint image, 1024×1024, 2 steps | FP32 | **18.56 s** | 20.03 s |
| SANA-Sprint image, 1024×1024, 2 steps | FP16 | **17.51 s** | 18.38 s |
| SANA-Sprint image, 1024×1024, 2 steps | BF16 | **16.51 s** | 17.14 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | FP32 | **14.62 s** | 23.59 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | FP16¹ | **12.87 s** | 24.88 s |
| SANA-Video preview, 192×128, 5 frames, 1 step | BF16 | **13.74 s** | 14.64 s |

¹ Both FP16 video engines use the same range-preserving policy described below.
This row compares that policy, rather than unmodified Diffusers FP16 execution.

Timing includes stage weight loading, prompt encoding, denoising and image/frame
decoding to RGB. It excludes application/runtime initialization, Python imports,
and final PNG/MP4 saving. The Python image probe also exports its small latent
for diagnosis inside its timed interval. Raw stage times and every process are
in [results.json](results.json).

## Matched inputs and reference

- Image: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`, revision
  `aa76e7f4f4928f378716b6716a2130fba3caf5b1`, 1024×1024, two steps, guidance 4.5,
  seed 42. Prompt: “A red fox in a snowy forest, soft morning light, detailed wildlife photograph”.
- Video: `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`, revision
  `db5f398b13ca086d09a50ce156c20527773841b1`, 192×128, five frames, one step,
  guidance 6, seed 42. Prompt: “A small red fox walks slowly through a sunny green
  meadow. Clear natural daylight, detailed fur, steady camera. motion score: 30.”
- Both engines receive identical prompt token IDs, masks and Java-generated
  FP32 starting noise; the image reference also receives the identical second
  sampler noise. IDs and noise hashes are in [inputs.json](inputs.json).
- Temurin 20.0.2; ten DeepJ executor threads, 2 GiB Java heap and 6 GiB Metal
  buffer limit. PyTorch 2.8.0, Diffusers 0.42.1 and Transformers 5.22.1 use ten
  CPU threads, one inter-op thread, inference mode and MPS CPU fallback disabled.
- Image Gemma encoding uses BF16 in both engines. The selected image precision
  applies to the denoiser and DC-AE decoder; sampler state remains FP32.
  Video precision applies to Gemma and the denoiser; sampler state and Wan
  frame decoding remain FP32.
- Python uses the official model classes and schedulers with an explicit staged
  loader. Its resident-stage loader transfers and synchronizes individual
  parameters. Its FP32 video hooks stream Gemma and denoiser blocks to fit this
  16 GiB machine. Both video engines batch positive/unconditional captions and
  guidance predictions for this small preview. Gemma attention is eager.
  These are comparisons with the supplied staged harness, not every possible
  loading, attention or offload configuration in Python.

## Implementation and correctness

Shared grouped/depthwise convolution has an optional no-grad Metal MPSGraph
implementation in FP32, FP16 and supported BF16 configurations. Gradient-enabled
convolution retains the differentiable tensor composition. Raw GPU layout copies
preserve complete storage bits; dense gathers avoid unnecessary coordinate work.

Bounded inference blocks reuse scratch and retain their result on the device.
Channel-first normalization preserves its storage layout. Diffusion compact RMS
affine operations now round at the same intermediate points as the Python model;
compact SiLU reuses fused SwiGLU with a unit multiplier, rounding the completed
activation. Inference linear attention combines its value and normalization
products. The original gradient-enabled linear-attention path remains in use.

The new padded Gemma batch API keeps each sequence's causal/sliding mask
independent. Video prompt encoding batches both captions; small video guidance
batches both predictions, while larger grids keep sequential bounded execution.

A final repeated gradient test caught a pre-existing race in compact Metal
normalization: scratch could be reused before all threads read a reduction's
result. A threadgroup barrier now protects that read. Repeated centered forward
and backward comparisons pass for FP16 and BF16. FP32 single-thread normalization
keeps its existing arithmetic.

### FP16 video range policy

Unmodified Diffusers FP16 produced a non-finite latent for this full checkpoint
and input. Its 1e-15 attention epsilon cannot be represented in FP16; some Q/K
projections and gated residual products also exceed FP16's finite range.
Both comparison engines therefore use FP32 Q/K projection arithmetic through
RMS normalization, FP32 attention denominator accumulation and FP32 gated
residual accumulation. Q/K return to FP16 after normalization, and convolution
and other linear layers retain FP16 execution. The exact Python reference patch
and original non-finite failure log are included in the evidence archive.

### Numerical checks

Independent small PyTorch/Diffusers references cover compact Gemma, SANA
predictions, DC-AE, video blocks, gradients, layout views and the new convolution
path. Full-checkpoint outputs are finite and visually recognizable. The fox
below is the final FP32 output. Compact formats produce different image details;
full-checkpoint images are not claimed to be pixel-identical between engines.

RGB differences below use 0–255 channel levels, comparing the same input per dtype.
Video means include all five frames. These are observed differences, not a claim
that every full-pipeline difference has been isolated to a single rounding operation.

| Precision | Image mean difference | Image maximum | Video mean difference | Video maximum |
|---|---:|---:|---:|---:|
| FP32 | 0.59943 | 124 | 0.00007 | 1 |
| FP16 | 18.46924 | 235 | 0.03634 | 1 |
| BF16 | 18.56172 | 245 | 0.34023 | 7 |

![Final FP32 fox](fox-fp32.png)

## Training and build controls

FP32 Qwen, Loop and Prism controls compare the master baseline and optimized
candidate on CPU and Metal: all loss sequences, final parameter hashes and final
gradient hashes match exactly. The probe uses width 64, 32-token sequences,
20 warm-up and 11 measured steps. Ordered small Gemma CPU/Metal controls also
preserve identical loss sequences and final named parameter bytes; sorting the
parameter list is confined to that probe, keeping clipping reduction order fixed
across JVMs. Gemma gradient hashes were not captured by this control. Independent
Gemma gradient oracles pass. These controls precede the final compact-only SiLU
and reduction-race fixes; neither changes their FP32 model calculations.

The final actual-hardware `mvn clean verify -Pfull-coverage` passes
3,073 tests with zero failures/errors and 29 optional
skips. Java and native code-quality, duplication, frontend, autograd/Metal Java
coverage, native coverage and kernel-selection gates pass. The separate focused
Metal run passes 149 tests, including the repeated compact-gradient regression.
Coverage summaries, logs and training control arrays are retained in the evidence.

## Memory and scope

Every standalone process has a timeout, RSS and 64 MiB swap-out guard. None was
stopped by a guard; the largest system swap-out increase during a process was
0.0625 MiB. RSS is CPU process memory, not total GPU/system memory.
Photo analysis and Spotlight workers were temporarily paused with permission and
resumed at completion. The user's training remains stopped.

No long video was run: the 81-frame default and full-size video throughput are
outside this comparison. CPU media-generation speed was not measured here.
The FP32 training controls do not constitute FP16/BF16 training speed comparisons.

[results.json](results.json) records timings, numerical checks, memory guards,
coverage and artifact hashes. [evidence.tar.gz](evidence.tar.gz) contains the
probes, Python references, exact noise files, validation logs and source diff.
Python is used for independent measurements; DeepJ's runtime remains Java/native.
