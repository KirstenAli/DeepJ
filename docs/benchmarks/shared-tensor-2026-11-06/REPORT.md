# Shared tensor performance: bounded qualification

6 November 2026. Baseline: remote master
`177f476d84d4b98f3ee1212d825a55b86a746a8f`. Branch:
`feature/shared-tensor-performance`.

These are small synthetic FP32 workloads on one Apple M4 with 16 GiB unified
memory, macOS 15.5 and Temurin 20.0.2. They assess shared backend changes and
complete training steps. They do not establish performance for full pretrained
models, every dtype, other hardware, or long training convergence.

## Changes retained

- Direct half-split rotary positions, including backward, eliminate the gather /
  rotate / un-gather sequence. The primitive is shared; Qwen additionally shares
  its cosine/sine tables across layers within each forward call. The tables do
  not become a persistent weight cache. Existing interleaved rotary is retained.
- CPU batched BLAS pins arrays once for the whole batch. Compatible transposed
  left operands also use BLAS, including weight-gradient products.
- Dense CPU FP32 normalization has native forward and backward loops preserving
  Java's reduction order. Causal softmax uses vector exponentials with a Java
  fallback near ambiguous rounding boundaries and for nonfinite inputs. Native
  softmax is selected for 4,096–262,144 elements; larger matrices keep Java's
  parallel path because the serial native prototype regressed there.
- Metal uses a bounded native buffer cache after synchronous completion: up to
  128 buffers, 32 MiB total, 4 MiB each. Byte sizes are cached alongside handles
  so searching does not retain every candidate Objective-C object. Cache entries
  are cleared when all live buffers are released, and after budgeted releases
  or allocation rollback. `DEEPJ_METAL_DISABLE_BUFFER_REUSE=1` disables it.
- Shared device head layouts use views where possible; the CPU keeps its earlier
  dense fast path. A reshape that cannot be represented as a view still copies.
- Shared Metal fused causal attention supports FP32 128-wide heads and sequences
  of at least 32 tokens divisible by 32, including grouped-query backward. The
  existing 64-wide routing threshold remains 256. Unsupported cases retain their
  fallback. One parameterized shader implements both widths.

These changes preserve model parameter inventories, optimizer definitions,
checkpoint formats, training schedules, and dtype policy. BLAS and fused attention
can change FP32 reduction order; bit-identical training trajectories are not promised.

## Master versus branch

Two fresh JVMs per variant/model/backend, baseline-first then candidate-first.
Each JVM performs 100 warm-up iterations and 30 measured iterations. The table
reports the median of the two process medians. JVM options are `-Xmx1g -Xbatch
-XX:CompileThresholdScaling=0.1`; DeepJ uses ten executor threads.

Each model uses vocabulary 256, width 256, context limit 128 and a fixed 64-token
sequence. Qwen has two layers, four query/two KV heads, head width 128 and FFN 512.
Loop has one shared core layer executed twice, four query/two KV heads and FFN 512.
Prism has two layers, four heads, FFN 512, query rank 128 and KV rank 64.

Inference includes prompt processing plus four cached continuation steps, full
logit reads and a checksum. It is not a tokens-per-second decoding benchmark.
Training includes forward, cross entropy, backward, global clipping and AdamW
at learning rate 1e-4. Every iteration synchronizes and releases temporary
resources. Cleanup cadence is part of this workload; normal training can use a
different cadence. Model initialization and JVM startup are excluded.

| Backend | Model | Inference: master → branch (ms) | Training: master → branch (ms/step) | Training time reduction |
| --- | --- | ---: | ---: | ---: |
| CPU | Qwen | 6.046 → 4.132 | 16.947 → 13.029 | 23.1% |
| CPU | Loop | 3.653 → 3.236 | 11.902 → 10.463 | 12.1% |
| CPU | Prism | 3.296 → 2.880 | 10.923 → 9.683 | 11.3% |
| METAL | Qwen | 9.242 → 5.979 | 6.987 → 4.021 | 42.4% |
| METAL | Loop | 7.549 → 7.299 | 4.938 → 4.251 | 13.9% |
| METAL | Prism | 7.803 → 7.316 | 5.297 → 4.687 | 11.5% |

Every final paired training comparison improved. The smaller inference differences
should be read in the context of run-to-run noise. CPU records use the final
measured CPU path for these shapes; the subsequent large-softmax guard does not
change their dispatch. Final Metal records include the revised cache lookup.

## PyTorch reference

A separate FNN comparison uses 64 examples, widths 256 → 512 → 512 → 128, ReLU,
FP32 parameters, MSE and AdamW (1e-4, betas 0.9/0.999, epsilon 1e-8, decay 0.01).
Both engines use the exact initial arrays and fixed inputs exported by Java.
PyTorch 2.8.0 uses eager tensor operations and its default AdamW implementation;
no compilation or quantization is enabled. CPU thread count is ten. Each fresh
process has 50 warm-up updates and 20 measured updates, including loss readback
and GPU synchronization.

| Backend | DeepJ process medians (ms/step) | PyTorch process medians (ms/step) |
| --- | ---: | ---: |
| CPU | 1.526, 1.491 | 1.456, 1.445 |
| METAL | 2.257, 1.248 | 2.591, 2.967 |

CPU training was close but remained slower in this probe. DeepJ's two final Metal
training process medians were lower; the variation is large enough that this is
exploratory evidence, not a general claim of beating PyTorch. The initial CPU FNN
inference probe also remained slower (about 0.21 ms versus 0.066 ms). This comparison
is with PyTorch's CPU/MPS execution, not with Python language execution in isolation.

## Candidates measured but not enabled automatically

Packing three projection weights once helped isolated inference. Repacking them
through the existing differentiable `TensorJoin.concatenate` on every forward
was substantially slower, especially on CPU. In the 64-position probe, CPU
forward/backward rose from about 0.55 ms to 7.85 ms, and Metal from 1.51 ms to
1.76 ms. A persistent packed copy would consume additional model memory and must
be kept synchronized with optimizer updates. Automatic load-time packing is
therefore not included. An explicit packed-parameter layer would require a
separate API/checkpoint design rather than silently changing existing parameters.

Non-MPS Metal operations already share a compute encoder. MPS matrix products
require ending that encoder; removing those boundaries without replacing the
matrix implementation would be incorrect. The retained changes reduce copies
and use fused attention instead. The final normalization input is not forcibly
copied just to select a dense kernel, because the added allocation/copy has no
established benefit.

## Numerical and resource checks

Direct rotary forward and input gradients match the previous gather-based path
bit for bit in FP32, FP16 and BF16, including strided layouts. CPU normalization
forward statistics, output and backward match a scalar strided Java reference
bit for bit. Causal softmax checks preserve masking, exceptional values and Java
rounding. Batched/transposed BLAS products and derivatives are checked against
independent double-precision calculations, including offsets and broadcasting.

The new 128-wide fused attention forward and Q/K/V derivatives are checked against
an independent double-precision oracle at lengths 32, 64, 96 and 256, with several
query/KV head ratios. Existing 64-wide and compact attention tests are also run.
The matched CPU model inference checksums remain identical. Metal Loop and Prism
inference and training checksums remain identical; Qwen's fused path has expected
FP32 differences. Maximum loss-curve differences in the 130-update controls stay
below 1.3e-5. These controls do not establish long-run convergence equivalence.

Native buffer tests recycle equal-sized FP32/BF16 storage, check zero initialization
and uploads, and preserve live buffers across cache clearing. No pretrained model
or dataset is loaded by the performance probes.

`mvn -o -Pfull-coverage verify` passed on actual Metal: 2,802 tests, zero failures
or errors, 29 optional tests skipped. Full Java line/branch coverage was
94.13%/83.55%; Metal Java 96.89%/84.93%; autograd 99.65%/95.74%; native
97.22%/86.88%, with 100% native functions and Metal kernel selections. Checkstyle,
PMD, duplicate-code and native readability checks passed without relaxed thresholds.
An additional Java-only fallback run passed all 24 selected tests with
`-Ddeepj.cpu.javaOnly=true`.

## Evidence and reproduction

[Raw timings and loss curves](results.json) retain individual samples and process
results. [Evidence archive](evidence.tar.gz) contains the standalone benchmark
sources, invocation scripts and verification logs. Scripts retain the original
workspace paths; substitute the checkout and evidence directories when replaying.

Build the baseline and branch in separate worktrees, using the same JDK and native
libraries, then create each runtime jar with `jar cf runtime.jar -C target/classes .`.
Compile `QuickProbe.java`, `TrainingProbe.java`, `ProjectionProbe.java` and
`OperatorProbe.java` against the branch jar. Run each revision in a fresh JVM with
the flags above and `-Dprobe.warmups=100 -Dprobe.samples=30`. For example:

```bash
java -Xmx1g -Xbatch -XX:CompileThresholdScaling=0.1 \
  -Dprobe.warmups=100 -Dprobe.samples=30 \
  --enable-native-access=ALL-UNNAMED -cp runtime.jar:evidence \
  QuickProbe metal loop training
```

The PyTorch script consumes the `.input` file exported by `TrainingProbe`; this
ensures the same input/target/parameter arrays. Retain both run orders and all
samples rather than selecting the quickest run.
