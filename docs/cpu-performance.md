# CPU performance

DeepJ uses portable Java CPU kernels, with optional CPU matrix acceleration on
Apple Silicon and macOS 15 or newer. Large FLOAT32 products use Accelerate BLAS.
Compact products use SIMD with FLOAT32 accumulation and bounded conversion tiles.
Cached attention reads capacity-strided views rather than copying its full prefix.
See [backend selection](backends.md#cpu-matrix-execution).

## FLOAT32 full-model comparison

The updated CPU implementation was measured on the complete Qwen3-1.7B checkpoint
on the same Apple M4, using FP32 weights and activations in both engines. Each
processed the same 64-token prompt and 16 fixed continuation tokens with a KV
cache, batch size one, ten CPU workers and full final-position logit readback.
PyTorch 2.8.0 ran first, followed by DeepJ; each used seven warm-up trials and
three timed trials. Loading, tokenization, sampling and rendering were excluded.

| Engine | Median prompt processing (ms) | Median cached decode (tokens/s) |
| --- | ---: | ---: |
| DeepJ CPU | 334.83 | 12.17 |
| PyTorch CPU | 311.25 | 7.99 |

DeepJ prompt processing took about 7.6% longer in this short test. Cached decoding
was faster. These local measurements establish neither general CPU parity nor a
speed advantage across other inputs, precision formats or machines.

IntelliJ, Chrome and idle Zoom were closed for the comparison. Neither engine had
new system swap-outs during a timed trial. PyTorch loading and its first warm-up
coincided with 394 MiB of swap-outs; subsequent warm-ups and timed trials had none.
DeepJ had none throughout its process. The CPU reference loader releases each
checkpoint mapping after converting one tensor, avoiding the larger temporary
footprint of converting all mapped weights together. Its 455,808 checked logits
were bit-identical to the previous reference loader.

All 455,808 final-position logits from the prompt and first two continuation
steps were finite. Against PyTorch, the maximum absolute difference was 0.0001664
and RMS difference was 0.0000217. Against the original DeepJ CPU implementation,
those differences were 0.0000777 and 0.0000115. All three top-token choices agreed.
[Raw trials, memory measurements and numerical checks](benchmarks/cpu-fp32-layout-m4.json)
record this full-checkpoint comparison and the smaller diagnostic separately.

## Earlier Qwen3-1.7B measurements

The full-model measurements below were collected at commit `a9cc1bc7`, before
the additional prompt-processing improvements described later on this page.

Measured on an Apple M4 with 16 GiB memory and macOS 15.5, using Java 20.0.2,
PyTorch 2.8.0 and Transformers 4.51.3. The checkpoint was
`Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Both engines used batch size one, the same 64 prompt token IDs, a KV cache and
the same 16 fixed continuation token IDs. Each performed three warm-up passes
followed by three timed passes. Model loading, tokenization, sampling and text
rendering were excluded. Every step read all final-position logits into a CPU
FLOAT32 array. DeepJ and PyTorch used ten CPU workers; Accelerate's thread cap
was ten, and PyTorch inter-op parallelism was one. The engines ran sequentially
with IntelliJ closed.

| Weight format | DeepJ median decode (tokens/s) | PyTorch median decode (tokens/s) | DeepJ median prompt processing (ms) | PyTorch median prompt processing (ms) |
| --- | ---: | ---: | ---: | ---: |
| FLOAT32 | 8.99 | 9.00 | 595 | 293 |
| FLOAT16 | 16.60 | 20.66 | 1,039 | 689 |
| BFLOAT16 | 16.38 | 20.04 | 1,026 | 1,009 |

FLOAT32 decode throughput is close in this short test. The compact paths remain
slower than PyTorch, and FLOAT32/FLOAT16 prompt processing remains slower. These
results do not establish parity across all CPU operations, models or machines.
The earlier unoptimized DeepJ FLOAT32 baseline, commit `82c68300`, measured
1.95 tokens/s on the same inputs with one warm-up and three timed passes.

DeepJ had no new swap-outs during the reported runs. The PyTorch FLOAT32 process
coincided with 344 MiB of system swap-outs; the process-wide measurement does not
locate these within loading, warm-up or timed execution. The compact reference
runs had no new swap-outs. These are local measurements, not a controlled hardware
performance study. [Raw trials and process measurements](benchmarks/cpu-qwen3-m4.json)
include the warm-up results and timing ranges.

## Precision

Matching weight formats does not make all intermediate calculations identical.
DeepJ's existing embedding implementation promotes activations to FLOAT32.
Mixed FLOAT32 activations and compact weights therefore retain FLOAT32 outputs
under the existing promotion rules. The PyTorch compact reference also uses
compact activations, with its model's internal FP32 normalization and positional
calculations. No new autocast policy is applied to DeepJ.

BLAS, fused multiply-add and vector reductions can change final FLOAT32 rounding.
The full-checkpoint check compared all 455,808 logits from a 64-token prompt and
two fixed continuation steps against the original DeepJ CPU implementation.

| Weight format | Maximum absolute logit difference | RMS logit difference | Same top token at each checked position |
| --- | ---: | ---: | --- |
| FLOAT32 | 0.0000777 | 0.0000112 | Yes |
| FLOAT16 | 0.0000992 | 0.0000134 | Yes |
| BFLOAT16 | 0.0000777 | 0.0000115 | Yes |

All compared logits were finite. Against the independent PyTorch FLOAT32 output,
DeepJ's maximum difference decreased from 0.000231 to 0.000166, and RMS difference
decreased from 0.0000294 to 0.0000224. Top tokens agreed at all three checked
positions. These checks complement scalar/double-precision matrix references,
gradient tests, cache tests, mutation tests and optimizer comparisons; they do not
claim bit-identical generation for every input.

## Initial prompt processing improvements

CPU last-axis selection resolves outer tensor strides once per row and copies
the original storage bits. Matrix causal softmax combines scaling, masking and
normalization without intermediate tensors or repeated exponential calculations,
preserving the existing mask values and FLOAT32 reduction order. Dense pointwise
work uses dynamically balanced blocks.

Compact matrix products with 2–64 rows and an inner dimension of at least 2,048
use 128-column conversion tiles instead of 512-column tiles. This reduces the
temporary converted weight buffer per tile by up to 75%. Larger row counts and
cached vector-product tile sizes retain their previous settings. Weight storage,
activation precision, accumulation precision and the native CPU kernels are
unchanged.

A bounded diagnostic retained one released Qwen3 decoder block at its original
width of 2,048 and feed-forward width of 6,144. Its tied embedding/output table
contained only the 25 vocabulary rows used by the fixed inputs. Both versions
processed the same 64-token prompt and 16 fixed continuation tokens, with ten CPU
workers, 40 warm-up trials and nine measured trials per process. Loading and
sampling were excluded; each forward read back the final-position logits.

| CPU weight format | Before prompt processing (ms) | After prompt processing (ms) | Reduction in time |
| --- | ---: | ---: | ---: |
| FLOAT32 | 20.64 | 15.77 | 24% |
| FLOAT16 | 33.04 | 23.17 | 30% |
| BFLOAT16 | 32.58 | 22.67 | 30% |

The baseline was `a9cc1bc7`. FLOAT32 ran in before/after/after/before order;
compact formats each ran before/after. Processes ran sequentially. The same
backend observation wrapper was disabled during timing in both versions; five
separate profiled prompts and first cached steps supplied the accuracy checks.
All checked logits retained identical bits across the versions for every format.
Cached decode timings also improved slightly in this probe.

These are measurements of a reduced diagnostic model, not full-model throughput
or a new comparison with PyTorch. The earlier full-model results remain historical
measurements. [Raw prompt/decode trials and validation results](benchmarks/cpu-prefill-m4.json)
include timing controls and source fingerprints.

Small deterministic Loop, Prism and Qwen training comparisons on CPU and Metal
retained identical losses, gradients and updated parameter values. The complete
Metal/desktop verification passed with 2,766 tests, zero failures and 29 optional
skips, including style, duplication and coverage gates. Metal sources and native
libraries, autograd rules, model implementations, optimizers and training-loop
sources were not changed by these prompt-processing updates.

## Native activation and storage reuse

The next CPU update uses native activation loops and direct storage access,
following the approach used by [PyTorch's activation kernels](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/Activation.cpp)
and [compact matrix-vector kernels](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp).
DeepJ retains its existing activation and accumulation precision.

On supported Apple CPUs, contiguous FLOAT32 SwiGLU uses double-precision vector
exponentials and the existing FLOAT32 arithmetic order. A conservative guard
retains the Java path near FLOAT32 rounding boundaries and for exceptional
exponentials. Small or aliased destinations also use Java. Matrices and
higher-rank tensors share the same CPU SwiGLU implementation, including its
existing backward calculation.

Compact products read contiguous FLOAT32 inputs directly, avoiding repeated
activation-buffer allocation, zeroing and copying. Strided or compact inputs
retain their packing path. Contiguous index selections copy complete trailing
blocks, and large normalization inputs distribute independent rows across CPU
workers without changing each row's reduction order. These paths keep no
expanded model-weight cache.

The same reduced Qwen diagnostic was measured again, with 40 warm-up trials and
nine timed trials per process. The preceding CPU implementation and the updated
implementation each ran twice; PyTorch 2.8.0 ran once per format. Each read back
the final-position logits after the prompt and every fixed continuation step.
All engines used ten CPU workers; PyTorch inter-op parallelism was one and
Accelerate's thread cap was ten. Processes ran sequentially.

| Weights | Previous DeepJ prompt (ms) | Updated DeepJ prompt (ms) | PyTorch prompt (ms) | Previous DeepJ 16 decode steps (ms) | Updated DeepJ 16 decode steps (ms) | PyTorch 16 decode steps (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOAT32 | 16.09 | 13.93 | 10.74 | 56.46 | 53.89 | 56.22 |
| FLOAT16 | 24.73 | 15.11 | 24.62 | 31.45 | 28.64 | 26.42 |
| BFLOAT16 | 24.82 | 15.36 | 36.87 | 30.47 | 27.98 | 27.44 |

Prompt time decreased by about 13% for FLOAT32 and 38–39% for compact weights.
Cached decode time decreased by about 5–9%. FLOAT32 prompt processing remains
slower than PyTorch in this probe. FLOAT16 decode still takes about 8% longer;
BFLOAT16 decode medians are close, with overlapping observed timing ranges.
The compact engines use different intermediate precision policies, as described
above. These reduced-model results do not replace full-checkpoint measurements.
[Raw trials, timing ranges and source fingerprints](benchmarks/cpu-native-m4.json)
record the preceding implementation and the updated source.

The sampled prompt and cached-step logits retained identical bits for all three
weight formats. A separate 1,048,576-value activation check covers finite,
subnormal, special and rounding-boundary values, with explicit accelerated and
fallback cases. Small Loop, Prism and Qwen training comparisons retained identical
losses, gradients and updated weights on CPU and Metal.

The complete Metal/desktop verification passed with 2,775 tests, zero failures
and 29 optional skips. A further 25 targeted tests passed with native CPU execution
disabled. Metal code, autograd rules, optimizers, models and training-loop sources
are unchanged. A runtime audit found no calls to the modified CPU helpers during
the measured Metal training steps for the three small models; its CPU positive
control recorded calls.

## FLOAT32 checkpoint layout

PyTorch's [linear implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Linear.cpp)
multiplies inputs by a transposed weight view. DeepJ's CPU checkpoint allocation
now stores large FLOAT32 projections in column-major order, allowing BLAS to use
the corresponding contiguous inner dimension. The existing compact-weight layout
policy is reused; this retains one allocation per weight and adds no expanded cache.
Tall embedding tables retain their row-major layout.

`ParameterFactory.forLoading()` uses a backend allocation hook for checkpoint
slots. Its default delegates to the existing zero allocator; only `CpuBackend`
overrides it. Ordinary FLOAT32 zero tensors, gradient buffers and randomly
initialized parameters retain their previous layout. Checkpoint values, shapes,
precision and optimizer equations stay the same. Matrix RMS normalization also
reuses the existing CPU normalization kernel, avoiding intermediate tensors while
preserving its original FLOAT32 operation order and saved backward statistics.

The same one-block diagnostic was measured in before/after/PyTorch/after/before
order, with 40 warm-ups and nine timed trials per process. All weights and
activations were FLOAT32.

| Engine | Median prompt processing (ms) | Median 16 cached decode steps (ms) |
| --- | ---: | ---: |
| Previous DeepJ | 14.21 | 53.75 |
| Updated DeepJ | 11.70 | 40.88 |
| PyTorch | 10.73 | 56.55 |

In this reduced diagnostic, updated DeepJ prompt processing took about 9% longer
than PyTorch, down from about 32% longer. Cached decode time also decreased.
The sampled prompt and first-step logits changed by at most 0.0000145 versus the
preceding DeepJ implementation, with the same top token. A different BLAS weight
layout can change FLOAT32 rounding; these results do not claim bit identity.

Checkpoint restoration and AdamW comparisons passed for all storage formats.
Large loaded FLOAT32 weights retain reusable contiguous FLOAT32 gradient buffers.
Small Loop, Prism and Qwen training checks retained identical losses, gradients
and updated weights on CPU and Metal. Full Metal/desktop verification passed
with 2,780 tests, zero failures and 29 optional skips; 14 additional checks passed
with native CPU execution disabled. [Raw measurements and source fingerprints](benchmarks/cpu-fp32-layout-m4.json)
record this update separately from the preceding measurements.

## Reproducing the comparison

Compile the test harness with `mvn test-compile`. Set `MODEL` to the local official
Qwen3-1.7B snapshot. The committed input file fixes the token IDs for both engines.
Run one engine at a time, with enough free memory for the selected weight format.

```bash
VECLIB_MAXIMUM_THREADS=10 java -Xmx10g --enable-native-access=ALL-UNNAMED \
  -cp target/classes:target/test-classes \
  io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Benchmark \
  "$MODEL" cpu FLOAT32 src/test/resources/qwen3/cpu/benchmark-input.json 10 7 3
```

In an environment with PyTorch 2.8.0, Transformers 4.51.3 and safetensors:

```bash
VECLIB_MAXIMUM_THREADS=10 python src/test/python/qwen3_cpu_benchmark.py \
  "$MODEL" FLOAT32 src/test/resources/qwen3/cpu/benchmark-input.json 10 7 3
```

Replace `FLOAT32` with `FLOAT16` or `BFLOAT16` for compact weights. The Java
benchmark keeps its original one-warm-up/three-run behavior when the final two
arguments are omitted. The reference retains FP32 rotary-frequency buffers.

The optional `Qwen3CpuLogits` harness writes every logit for the three accuracy
contexts. `CpuMatmulBenchmark` measures individual matrix shapes. Neither harness
runs automatically or downloads model weights.

## Change scope

Production changes concern CPU execution, a separate CPU native library and the
checkpoint allocation hook described above. Metal sources, the packaged Metal
library, the shared executor, autograd, models and training-loop sources are
unchanged. Optimizer equations are unchanged. CPU training uses the updated CPU
matrix kernels and existing gradient rules. The native source checker
also analyzes the ARM CPU source for its target architecture on Intel Macs.
