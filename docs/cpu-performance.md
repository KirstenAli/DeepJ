# CPU performance

DeepJ uses portable Java CPU kernels, with optional CPU matrix acceleration on
Apple Silicon and macOS 15 or newer. Large FLOAT32 products use Accelerate BLAS.
Compact products use SIMD with FLOAT32 accumulation and bounded conversion tiles.
Cached attention reads capacity-strided views rather than copying its full prefix.
See [backend selection](backends.md#cpu-matrix-execution).

## Qwen3-1.7B measurements

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

## Reproducing the comparison

Compile the test harness with `mvn test-compile`. Set `MODEL` to the local official
Qwen3-1.7B snapshot. The committed input file fixes the token IDs for both engines.
Run one engine at a time, with enough free memory for the selected weight format.

```bash
VECLIB_MAXIMUM_THREADS=10 java -Xmx10g --enable-native-access=ALL-UNNAMED \
  -cp target/classes:target/test-classes \
  io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Benchmark \
  "$MODEL" cpu FLOAT32 src/test/resources/qwen3/cpu/benchmark-input.json 10 3 3
```

In an environment with PyTorch 2.8.0, Transformers 4.51.3 and safetensors:

```bash
VECLIB_MAXIMUM_THREADS=10 python src/test/python/qwen3_cpu_benchmark.py \
  "$MODEL" FLOAT32 src/test/resources/qwen3/cpu/benchmark-input.json 10 3 3
```

Replace `FLOAT32` with `FLOAT16` or `BFLOAT16` for compact weights. The Java
benchmark keeps its original one-warm-up/three-run behavior when the final two
arguments are omitted. The reference retains FP32 rotary-frequency buffers.

The optional `Qwen3CpuLogits` harness writes every logit for the three accuracy
contexts. `CpuMatmulBenchmark` measures individual matrix shapes. Neither harness
runs automatically or downloads model weights.

## Change scope

Production changes are confined to CPU execution and a separate CPU native library.
Metal sources, the packaged Metal library, the shared executor, autograd, models,
optimizers and training-loop sources are unchanged. CPU training uses the same
new CPU matrix kernels and existing gradient rules. The native source checker
also analyzes the ARM CPU source for its target architecture on Intel Macs.
