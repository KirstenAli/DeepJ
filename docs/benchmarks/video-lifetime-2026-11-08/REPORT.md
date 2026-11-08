# Video sampler and compact precision follow-up

The follow-up to the [photo sampler fix](../photo-sampler-2026-11-08/REPORT.md)
checks video for the same lost-input lifetime error. Video already retains its
FP32 latent across denoiser cleanup and retains the solver's previous prediction.
The new regressions exercise both behaviors, including the second-order midpoint
step. No DeepJ runtime, training, shared tensor or native kernel changes were needed.

## Correcting the Python reference

The initial three-step pretrained FP32 comparison averaged 9.77648 RGB channel
levels of difference. A small independent reproduction isolated an MPS reference
problem: on this Apple M4 with PyTorch 2.8.0, multiplying an MPS tensor by
`torch.tensor([0.1, 0.2, 0.3])[1]` used 0.1 instead of the view's 0.2.
Diffusers' scheduler indexes CPU sigma tensors in this way. A fixed-flow
three-step CPU/MPS scheduler control diverged by maxima 0, 0.00914848 and
0.20239711. Keeping scheduler arithmetic on CPU produced zero difference at every
step. [`sana-video-mps-scheduler-control.py`](../../../scripts/sana-video-mps-scheduler-control.py)
reproduces the issue without pretrained weights.

The corrected reference runs the small FP32 scheduler on CPU, transfers its result
back to MPS and explicitly copies each model timestep to MPS. Model inference and
Wan decoding stay on MPS. This follows the CPU-scheduler policy already used by
the earlier multi-step video oracle. The one-step comparison does not exercise
the failing later sigma views.

## Complete pretrained short previews

Both engines use the same checkpoint, saved prompt token IDs and masks, Java seed-42
noise, 192×128 output, five frames, three steps and guidance 6. The fox/meadow prompt
is the one used by the earlier preview controls. Each engine computes its own
Gemma features, model predictions, feedback trajectory and rendered frames.
Three steps exercise midpoint history while keeping the test short.

Apple M4, 16 GiB unified memory, macOS 15.5, Java 20, PyTorch 2.8.0,
Transformers 5.22.1 and Diffusers 0.42.1. The selected dtype applies to Gemma and
the denoiser; the sampler and Wan decoder use FP32. Generated rotary buffers keep
their constructor precision. FP16 uses the same previously documented
range-preserving Q/K, denominator and gated-residual policy in both engines;
that FP16 reference includes explicit stability adjustments to upstream Diffusers.

| Precision | Mean RGB difference | Maximum RGB difference | 99th percentile |
| --- | --- | --- | --- |
| FP32 | 0.00110 | 1 | 0 |
| FP16 | 0.60009 | 11 | 4 |
| BF16 | 5.56881 | 78 | 33 |

All values are absolute channel differences on the 0–255 scale, over all pixels
and all five frames. Compact output is not pixel-identical. These very short,
low-resolution previews check numerical behavior; they do not establish the
quality or memory requirements of 81-frame generation.

## Same-precision Python rounding control

A second complete Python run separates `Linear` matrix multiplication and bias
addition, matching the intermediate rounding point of DeepJ's composition.
The checkpoint, dtype, IDs, masks, noise, scheduler and decoder policy stay fixed.
This is a deliberate Python arithmetic variant, separate from the baseline above.

| Precision | Python baseline versus rounding control: mean / max RGB | DeepJ versus rounding control: mean / max RGB |
| --- | --- | --- |
| FP16 | 0.86838 / 20 | 0.41751 / 9 |
| BF16 | 4.81024 / 71 | 2.62212 / 47 |

Changing rounding points in Python alone reproduces differences of comparable
scale. Matching that composition reduces the DeepJ gap, establishing rounding
sensitivity as a contributor. The independent component, same-input prediction,
sampler-history and CPU/Metal frame checks found no remaining incorrect video
operation or disappearing-input defect. These checks support numerical closeness
for the tested configurations, without asserting exact agreement at other sizes.

The tiny nine-frame, three-step pipeline uses an independent CPU Python oracle in
FP32, FP16 and BF16. Its complete CPU-rendered frame differences averaged 0.000145,
0.04933 and 0.21962 RGB levels, with maxima 1, 1 and 10. The Python-only tiny
rounding control itself reaches maximum 28 and mean 0.50116 in BF16. This informs
the BF16 frame regression's rounding allowance. Sampler-only comparisons use the
exact Python flow predictions after actual denoiser cleanup and require agreement
within 0.000002 at every step, independently of model rounding.

## Resource and timing observations

Every accepted preview and rounding-control run completed with no additional
swap. Two earlier compact Python attempts were stopped by the swap-growth guard
before producing output; they supply no accepted timing or quality result.
IntelliJ and Chrome were closed for the accepted controls. Python Gemma weights
are streamed to bound memory; the compact denoiser is cached and FP32 streams.

These are single correctness runs, rather than repeated performance estimates.
Generation timings are retained in [results.json](results.json): FP32 took
24.32 s in DeepJ and 37.73 s in Python; FP16 took 19.99 s and 25.79 s; BF16 took
20.07 s and 17.75 s. The BF16 three-step observation favors Python. It does not
establish an across-workload speed claim or replace the earlier one-step timings.
This follow-up changes no DeepJ execution code.

## Regression coverage

- `CompactVideoReferenceTest` compares full nine-frame rendered output to Python
  and isolates three-step sampler history after compact denoiser cleanup.
- `VideoGuidanceTest` additionally forces sequential guidance, streamed weights
  and the staged attention path with over 8,192 latent tokens in every dtype.
- The existing photo CPU/Metal sampler regression is rerun in all three dtypes.
- Existing FP32 20- and 50-step tiny trajectory/decoder oracles are retained.

Generate the new small oracle with
`python scripts/sana-video-compact-pipeline-reference.py`. Add `--split-bias` to
generate the separate Python rounding control. The pinned reference environment
is in [requirements.txt](requirements.txt); Python is used only for test references.

Verification commands on the physical Mac:

```sh
mvn test -Dsanavideo.test.metal=true \
  -Dtest=CompactVideoReferenceTest,VideoGuidanceTest,CompactMediaPipelineTest,SanaVideoTrajectoryTest,SanaVideoEndToEndReferenceTest
mvn clean verify -Pfull-coverage
```
