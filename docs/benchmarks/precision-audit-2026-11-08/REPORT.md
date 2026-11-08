# Precision and default-chat verification

Checks on an Apple M4 with 16 GiB unified memory, macOS 15.5, Java 20,
PyTorch 2.8.0, Transformers 5.22.1 and Diffusers 0.42.1. The source baseline is
`5b67a0bc`; the changes select checkpoint precision for Qwen chat, reduce CPU
checkpoint-loading memory, and restore SANA's FP16 overflow clamp.
Raw numerical summaries and Qwen timed trials are in [results.json](results.json).

## Qwen chat default

The official Qwen3-1.7B files store BF16 parameters. Automatic chat loading now
uses BF16 weights and embedding activations, matching the existing explicit
`--dtype bfloat16` path. Explicit FP16 and FP32 overrides remain available.
New-model construction and the lower-level pretrained-loader activation policy
retain their existing behavior; this change does not set a training precision.

Both engines used the same checkpoint, batch size one, a KV cache, a 64-token
prompt and 16 fixed continuation tokens, with full logit readback and no sampling.
Each engine ran twice in fresh processes, in reversed order, with two warm-ups
and three timed trials per process. Figures are medians of six timed trials;
both engines used ten CPU threads.

| Backend | DeepJ decode | PyTorch decode | DeepJ prompt | PyTorch prompt |
| --- | --- | --- | --- | --- |
| CPU | 21.46 tokens/s | 18.14 tokens/s | 744.1 ms | 963.1 ms |
| Metal / MPS | 23.60 tokens/s | 23.34 tokens/s | 105.53 ms | 105.13 ms |

The small Metal decode lead is an observed result, not a statistical significance
claim. Metal prompt times were effectively tied. These measurements are separate
from the older runs in the performance guide.

The CPU importer previously staged a complete source tensor beside its target;
the full chat checkpoint exhausted even an 8 GiB heap during this audit. Matching
CPU storage now restores directly into the target, including transposed views.
The full default model ran with a 6 GiB heap, approximately 5.5 GiB peak RSS and
no additional swap. Device transfers and explicit dtype conversions retain their
previous staging path. Tests verify raw storage bits, logits, gradients and
checkpoint round trips.

## FP32 image and video correctness

The saved 1024×1024, two-step SANA-Sprint fox comparison has a mean absolute RGB
difference of **0.59943 channel levels out of 255**, with an isolated maximum of
124. It is not pixel-identical. To investigate the difference, the pretrained
denoiser was checked with identical saved inputs at every one of its 28 blocks,
using an 8×8 latent grid and 32 caption tokens. The largest block relative RMSE
was **2.424e-7**. The complete small denoiser's relative RMSE was **4.027e-6**;
PyTorch CPU versus PyTorch MPS differed by **7.570e-6** on the same input.

The actual pretrained image decoder, given identical 2×2 latents, differed by a
mean **6.919e-8** and maximum **5.364e-7** in normalized pixels. These stage
controls found no FP32 arithmetic or layout mismatch explaining the image result.
They support accumulated floating-point differences rather than a malformed
operation. Maximum full-image RGB differences should not be described as one
colour level.

A new FP32 video control used identical token IDs, masks and noise at 192×128,
five frames and one sampling step. Its mean RGB difference was **0.000071** and
its maximum was **one channel level**. It also completed MP4 export. The earlier
[nine-frame, full-resolution comparison](../../sana-video.md#pretrained-comparison)
remains the longer FP32 quality check; no long video run was repeated here.

## Compact image and video checks

Each pretrained image block was also run with identical compact reference
inputs. The largest block relative RMSE was **0.0742% for FP16** and **0.4290%
for BF16**. Differences accumulate across blocks: the complete small denoiser
differed by **5.13%** and **11.08%**, respectively. PyTorch CPU versus MPS also
diverged in these formats: **0.78%** and **7.97%**. The isolated decoder's mean
normalized-pixel differences were **0.000111** and **0.000906**.

The full compact foxes remain visibly different from their same-dtype Python
references. Local block agreement does not imply matching full images. The
FP32 result and the compact result should therefore be described separately.

One concrete port omission was found: Diffusers clips SANA self-attention's
FP16 output projection to `[-65504, 65504]`. DeepJ now applies that same guard.
A regression forces projection overflow and checks the finite result on CPU
and Metal. FP32/BF16 and finite FP16 fox pixels are unchanged by the clamp.

## Short generation timings and output differences

These are single fresh-process controls, not stable performance estimates.
Timing includes stage loading, prompt encoding, sampling and RGB decoding;
it excludes runtime startup, Python imports and file export. Image controls use
1024×1024, two steps; video controls use 192×128, five frames and one step.
Each pair receives identical saved inputs. All guarded runs completed without
additional swap. The FP32 photo measurement precedes the compact-only clamp;
FP32 execution is unchanged.

| Workload | Precision | DeepJ | Python MPS | Mean RGB difference | Maximum RGB difference |
| --- | --- | --- | --- | --- | --- |
| Photo | FP32 | 17.51 s | 19.46 s | 0.59943 | 124 |
| Photo | FP16 | 17.39 s | 18.53 s | 18.46924 | 235 |
| Photo | BF16 | 16.75 s | 17.14 s | 18.56173 | 245 |
| Video preview | FP32 | 15.13 s | 22.15 s | 0.000071 | 1 |
| Video preview | FP16 | 12.82 s | 23.62 s | 0.03634 | 1 |
| Video preview | BF16 | 13.63 s | 15.36 s | 0.34023 | 7 |

RGB differences use the 0–255 channel scale and include every video frame.
Image Gemma encoding uses BF16; the image denoiser and DC-AE use the selected
precision. Video precision applies to Gemma and the denoiser; the sampler and
Wan decoder retain FP32. FP16 video uses the same range-preserving adjustments
in both engines, rather than unmodified upstream FP16. The staged Python loader,
offload and precision policies are described in the
[original matched controls](../media-throughput-2026-11-07/REPORT.md#matched-inputs-and-reference).
These previews do not validate 81-frame generation or its quality.

## Build and regression checks

`mvn clean verify -Pfull-coverage` passed from IntelliJ's terminal with physical
Metal access: 3,102 tests counted, 3,073 passed, 29 optional checkpoint, diagnostic
or model-specific tests skipped. Java/native coverage gates, all Metal kernel
selections, Checkstyle, PMD, copy-paste detection and UI linters passed.
Separate focused Metal tests exercised the new default Qwen dtype checks and
the SANA compact reference/overflow controls. Javadocs were regenerated.

Loop, Prism, optimizers, shared tensor operations and native kernels are unchanged
in this patch. The SANA change affects FP16 overflow handling; the Qwen default
change is in chat loading. CPU import tests preserve checkpoint values and
autograd behavior after direct restoration.
