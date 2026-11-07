# Shared CPU training optimizations

Measured on 7 November 2026, Apple M4, 16 GiB, macOS 15.5,
Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.

The same [small-Qwen training workload](../qwen-training-2026-11-07/REPORT.md#matched-workload-and-timing)
now measures **6.633 ms/step in DeepJ versus 9.080 ms/step in PyTorch**.
These are medians of two fresh-process medians, with engine order reversed
for the second pair. This is the 1,640,192-parameter training configuration,
not training the full Qwen3-1.7B checkpoint.

| Engine | First process | Second process | Summary |
| --- | ---: | ---: | ---: |
| DeepJ CPU | 6.646 ms | 6.621 ms | **6.633 ms** |
| PyTorch CPU | 9.102 ms | 9.057 ms | 9.080 ms |

The milestone JAR used the packaged native CPU library from before the subsequent
rebuild. These timings therefore establish the Java-side improvements; they do not
measure the native row-parallel change. See the [follow-up measurements](../compact-cpu-metal-2026-11-07/REPORT.md)
for the rebuilt native libraries and compact execution work.

## Implementation

- Dense FP32 cross-entropy shares the computed exponentials between loss and gradient.
- RMSNorm backward reuses the existing fused normalization implementation.
- Contiguous FP32 additions use a direct loop, with parallel work reserved for larger buffers.

These are shared tensor/backend changes. No model-specific shortcut, optimizer
change, or change to the loss, gradient-clipping policy or batch is involved.
Broadcasting, overlapping views, compact storage and the Java-only fallback remain covered.

## Before/after controls

Each process used 100 warm-up updates and 30 measured updates. All three models
retained **identical complete loss sequences and final parameter/gradient SHA-256
hashes** before and after, on both CPU and Metal.

| Model | CPU before | CPU after | Metal before | Metal after |
| --- | ---: | ---: | ---: | ---: |
| Small Qwen | 11.438 ms | 6.625 ms | 4.088 ms | 4.163 ms |
| Loop | 10.451 ms | 5.325 ms | 4.446 ms | 4.484 ms |
| Prism | 9.585 ms | 5.843 ms | 4.569 ms | 4.594 ms |

The Metal controls use the same Metal implementation in both JARs. Small timing
movements in these short runs do not establish a performance change.
The new CPU results retain the previously documented numerical differences from
PyTorch, including the existing FP32 global-norm reduction; those calculations
were not changed for this optimization.

Full verification passed: 2,825 tests, no failures, 29 skips; Java and native
coverage, Checkstyle, PMD, duplication and native complexity checks passed.
The targeted Java-only fallback checks also passed.

## Reproduction

Use the [existing training benchmark launcher](../qwen-training-2026-11-07/run.sh)
with a release JAR containing these CPU changes. Preserve its FP32 fixture, thread
counts, JVM flags, 100 warm-up steps and 30 measured steps. Run engines sequentially,
with builds and other tests stopped. The matched runs here used the same exported
initial weights as the earlier comparison.

[results.json](results.json) records all timings, loss sequences, hashes and source
fingerprints. The earlier production baseline is `4ed97933`; the candidate CPU
sources are based on `e9142cbb`. This report records the CPU milestone separately
from subsequent FP16/BF16 Metal work.
