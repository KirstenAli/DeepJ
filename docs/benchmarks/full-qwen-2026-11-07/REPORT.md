# Full Qwen3-1.7B: CPU and Metal follow-up

Measured on 6–7 November 2026 on an Apple M4 with 16 GiB unified memory,
macOS 15.5, Temurin 20.0.2, PyTorch 2.8.0 and Transformers 4.51.3.
These are full pretrained-model measurements, separate from the small training
probes in [the shared tensor report](../shared-tensor-2026-11-06/REPORT.md).

## Results

Each entry is the median of five measured trials after eight warm-up trials in
one fresh process. Higher decoding throughput is better; lower prompt time is
better. PyTorch was measured again for this comparison.

| Measurement | DeepJ | PyTorch | Difference |
| --- | ---: | ---: | --- |
| CPU cached decoding | 12.72 tokens/s | 9.02 tokens/s | DeepJ 41.0% higher throughput |
| CPU prompt processing | 292.8 ms | 286.5 ms | DeepJ 2.2% longer |
| Metal / MPS cached decoding | 12.69 tokens/s | 12.05 tokens/s | DeepJ 5.3% higher throughput |
| Metal / MPS prompt processing | 119.7 ms | 132.3 ms | DeepJ 9.5% shorter |

DeepJ CPU prompt processing ranged from 289.9–299.9 ms; PyTorch CPU ranged from
284.4–288.3 ms. The remaining CPU prompt gap is real in these samples; it is not
reported as a tie. These short runs establish this FP32 workload, not performance
for every model, dtype, context length or hardware configuration.

DeepJ CPU was measured at `4ed97933`. Metal was measured at `d5baf672`, before
that CPU-only optimization follow-up; the measured Qwen Metal execution path is
unchanged by the follow-up. The earlier README numbers are historical results,
not the PyTorch reference used in this table.

## Matched workload

Both engines load checkpoint revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
from Qwen3-1.7B, expand its BF16 weights to FP32, and run batch size one. Both
process the same 64 input token IDs, then the same 16 fixed continuation token
IDs using a KV cache. There is no sampling, text tokenization or training inside
the timing. Each forward reads the full final-position logits back to the CPU;
that read synchronizes GPU work. Prompt processing and cached decoding are timed
separately. Model construction, weight loading and process startup are excluded.

DeepJ uses ten executor threads and `VECLIB_MAXIMUM_THREADS=10`. PyTorch uses ten
CPU threads, one inter-op thread and Transformers' `sdpa` attention implementation.
PyTorch uses eager execution without `torch.compile`; neither model is quantized.
JVM heap maximum is 10 GiB.
The complete inputs, samples, checksums and process metrics are in
[results.json](results.json).

## Follow-up optimizations

The full CPU profile exposed roughly 1.5 ms per block in generic half-split
rotary indexing. A shared dense FP32 implementation now handles both interleaved
and half-split rotary positions, including the inverse operation used by backward.
It supports contiguous slices with offsets and retains the typed/strided fallback.
Large row sets use the existing CPU executor; small operations run directly.

Native CPU causal softmax distributes independent rows across bounded workers.
The worker count respects the executor's thread limit and serial setting. Arithmetic
within each row, masking, and the rounding-boundary fallback remain unchanged.
These changes apply to shared tensor operations rather than a special Qwen-only
forward implementation. No Metal source or binary changed in this follow-up.

The same full CPU benchmark before this follow-up measured 336.3 ms prompt time
and 12.43 tokens/s decoding. Afterwards it measured 292.8 ms and 12.72 tokens/s:
a 12.9% reduction in prompt time. A full-width one-block probe measured
11.75 → 10.24 ms; all its captured logits remained bit-identical.

## Training and correctness controls

The small Qwen, Loop and Prism training controls use the same setup as the shared
tensor report: 100 warm-up updates and 30 measured updates, including backward,
clipping, AdamW and synchronization. Every follow-up comparison produced identical
loss sequences and final parameter/gradient hashes. This compares `4ed97933`
with `d5baf672`, not with the older master revision before the fused-attention work.

| Backend | Model | Before → after (ms/step) |
| --- | --- | ---: |
| CPU | Qwen | 12.857 → 11.500 |
| CPU | Loop | 10.361 → 10.379 |
| CPU | Prism | 9.642 → 9.688 |
| Metal | Qwen | 4.015 → 3.962 |
| Metal | Loop | 4.397 → 4.230 |
| Metal | Prism | 4.643 → 4.590 |

CPU entries combine two fresh processes per revision with reversed order. Qwen's
CPU training time fell 10.6%; Loop and Prism differed by less than 0.5%. Metal
entries are a single before/after control and do not establish an additional
Metal speedup from the CPU changes. These are bounded regression checks, not
full-size training-throughput claims.

Dense rotary forward/inverse results match an independent strided route exactly,
including offsets and parallel/serial execution. Existing gradient tests pass.
Softmax tests compare against scalar Java arithmetic, including masking, NaNs,
infinities and parallel execution. `mvn -o -Pfull-coverage verify` passed on actual
Metal: 2,807 tests, zero failures/errors, 29 optional skips. Style, duplication,
Java/native coverage and kernel-selection gates passed. An additional Java-only
fallback run passed all 45 selected tests.

## Resource handling and reproduction

Each full-model engine ran alone with IntelliJ and Chrome closed. A guard stopped
processes exceeding 10–11 GiB RSS, 128 MiB of system swap-out growth or the time limit.
Accepted DeepJ and PyTorch CPU runs had no swap-out growth. The accepted MPS run
had 11.7 MiB across the whole process. RSS omits some driver allocations and must
not be used to claim an engine memory advantage; MPS reported 6.88 GB of tensor
allocations and 8.96 GB of driver allocations immediately after loading.

Initial guarded attempts were excluded. The DeepJ benchmark originally copied all
GPU weights to Java during final resource release, after timing had finished. Its
standalone teardown now releases temporary resources and lets process exit reclaim
the model, avoiding that unnecessary download. This changes only the benchmark.
The PyTorch reference loader converts each BF16 tensor to FP32 on MPS directly and
releases unused loading allocations before timing; this avoids a large temporary
CPU FP32 staging allocation. Every measured weight is still FP32.

[The evidence archive](evidence.tar.gz) includes benchmark sources, launch scripts,
accepted and excluded logs, the fixed input and verification output. Substitute
local checkout, checkpoint and virtual-environment paths when replaying scripts.
Build the recorded DeepJ revisions separately; run each engine in a fresh process.
The optional Python files are reference/testing tools, not DeepJ runtime code.

The later [CPU softmax follow-up](../cpu-softmax-2026-11-07/REPORT.md) measures
lower prompt time in a full-width decoder block while preserving exact scalar
results. Its full-model retry is memory-limited, so it does not replace the
full-model measurements in this report.

A subsequent [completed full FP32 CPU prompt comparison](../full-cpu-prompt-2026-11-07/REPORT.md)
measures 283.7 ms for DeepJ and 294.0 ms for PyTorch. It replaces the README's
CPU prompt row; the other historical measurements above remain unchanged.
