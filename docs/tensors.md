# Tensors, shapes and precision

DeepJ uses one dense `Tensor` API for scalars, vectors, matrices and higher-rank
arrays. Existing matrices are rank-two tensors; values default to FLOAT32.

ND and optional-dtype APIs are included in `0.12.0-alpha` and newer.

```java
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

Tensor batch = new Tensor(2, 3, 4);          // [batch, time, width]
Tensor a = Tensor.from(new float[]{1.5f, 2.5f});
Tensor b = Tensor.from(new float[]{1.5f, 2.5f}, DType.FLOAT16);
Tensor c = Tensor.from(new float[]{1.5f, 2.5f}, DType.BFLOAT16);
```

Use `shape()`, `rank()`, `size(axis)` and `numel()` to inspect dimensions.
`get(indices...)` and `setAt(value, indices...)` access logical values, including
noncontiguous views. `toFloatArray()` returns an independent row-major FLOAT32 copy.
Negative axes count from the end. Rank-zero scalars and zero-size axes are supported;
element counts must fit DeepJ's dense allocation limits.

## Operations and views

Elementwise operations broadcast matching trailing dimensions: dimensions must be
equal or one must be 1. `matmul` supports vector, matrix and broadcasted batched
products; it multiplies the final matrix axes without mixing independent batches.
Use `sum(axis, keepDimensions)`, `mean(axis, keepDimensions)` and `softmax(axis)`
for axis-specific operations. `sumTensor()` keeps a total differentiable; `sum()`
extracts a Java number.

`view`, `permute`, `transpose(axis1, axis2)`, `select` and `slice` can share storage.
`view` requires compatible strides; `reshape` may copy when needed. `contiguous()`
may return the original tensor when already contiguous. `copy()` creates independent
storage while preserving gradient tracking. DeepJ's `detach()` also creates an
independent copy, without gradient tracking; it is not PyTorch's shared-storage detach.

Shared views share mutation checks. Overlapping expanded views cannot be written.
Use managed tensor operations, not raw array writes, while autograd graphs are live.
First-order gradients handle broadcasts, views and repeated indices; gradients for
FLOAT16/BFLOAT16 inputs are FLOAT32.

## Storage is not the same as calculation precision

FLOAT32 stores four bytes per value. FLOAT16 and BFLOAT16 genuinely store two, with
different ranges and precision. Select them explicitly or use `tensor.to(dtype)`.
For binary tensor operations, identical input dtypes stay in that dtype; differing dtypes promote to FLOAT32,
including FLOAT16 mixed with BFLOAT16. A FLOAT32 operand is never silently narrowed.

Supported Metal compact-matrix paths use native compact operands with FLOAT32
accumulation. Eligible larger FLOAT16 products use Apple MPS on supported Apple GPUs.
MPS accepts FLOAT16 on both sides or FLOAT32 on the left and FLOAT16 on the right;
it does not directly accept the reverse mixed order or BFLOAT16 operands.
For eligible products, DeepJ losslessly widens BFLOAT16 operands, or a FLOAT16 left
operand paired with FLOAT32, into scoped GPU buffers. Staging is used only when all
matrix dimensions are at least 64 and the widened element count times 128 is no more
than rows x inner x columns, so the cast stays small next to the arithmetic. Products
whose cast would dominate retain the compact kernel. Two BFLOAT16 inputs produce a
result rounded back to BFLOAT16.
These are per-product conversions, not persistent FLOAT32 parameter copies.
FLOAT32 operands are never narrowed, and FLOAT32-only routing is unchanged.
Other shapes, layouts and device capabilities retain their appropriate GPU paths.
Other operations and unsupported native compact-matrix paths may use
FLOAT32 arithmetic on the selected backend. This is not automatic mixed precision:
there is no implicit autocast or loss scaling, and no hidden CPU fallback from Metal.
Optimizer master weights, gradients and moments remain FLOAT32. Consequently, two-byte
weights do not halve total training memory or guarantee faster training.

`ParameterFactory(dtype)` applies the requested compact dtype to parameters with more
than one row. One-row parameters, including norm gains and biases, stay FLOAT32.
Embedding lookups widen to FLOAT32, keeping the activation stream eligible for fused
FLOAT32 kernels while larger weight matrices retain two-byte storage. This is an explicit
parameter policy, not automatic autocast. A descriptor-based factory and loaded
checkpoints preserve their individual stored dtypes.

General compact, strided and broadcast CPU matrix products stage their operands in
temporary contiguous FLOAT32 arrays before the parallel product, then round the result
to its output dtype. This avoids repeated address calculations and compact-value decoding
inside every multiply, at the cost of temporary host memory, including expanded batches.
The dtype-converting contiguous copy also runs in parallel. These CPU products use
FLOAT32 arithmetic, so compact storage adds decoding/encoding work rather than native
16-bit arithmetic. The existing dense FLOAT32 matrix fast path remains separate.

Relative dtype speed depends on the operation, shape, backend and device. Short manual
timing probes are exploratory, not a guarantee that compact training matches or exceeds
FLOAT32 speed. The earlier [compact-precision measurements](benchmarks/compact-attention-2026-10-31/REPORT.md)
retain their original runtime and workload scope.

A historical short CPU probe on an Apple M4 MacBook Air reported roughly 300-330 ms
per FLOAT32 step and 350-370 ms for FLOAT16/BFLOAT16; compact steps before staging
were reported near 20 seconds. It used a two-layer Origin model, width 384,
256 tokens and a 4,096-token vocabulary. That probe used two warm-up steps, four
measured steps, fixed dtype order and explicit GC inside the timed interval. These
figures are exploratory, not performance qualification or proof of a consistent
15% dtype gap. Compact weights use less storage; total memory and speed need
matched measurements for the actual workload.

`MetalBackend.supportsNative16Matmul(dtype)` reports native compact-matrix support
on the active device. It does not promise that every operation uses compact arithmetic.

### Fused attention

Supported Metal devices can use fused causal attention with same-dtype FLOAT32,
FLOAT16 or BFLOAT16 inputs, 64-wide heads and complete 32-token tiles. Automatic
selection starts at 256 tokens; other shapes retain the separate GPU operations.
Compact inputs are widened only in SIMD register tiles. Fused-attention arithmetic,
normalization, accumulation and gradients are FP32; the returned output is rounded to its input dtype.
An unrounded FP32 output is saved for backward, avoiding gradient errors from using
the rounded compact output. Saved storage grows linearly with sequence length rather
than storing the complete attention-score and probability matrices.

Unlike separate compact operations, fused attention does not round scores and
probabilities into 16-bit storage between operations. Compact results and training
trajectories can therefore differ from the separate path; this does not change the
default FLOAT32 computation or enable automatic mixed precision. Actual speed and
memory gains depend on the workload and device.

`storageBytes()` reports the shared allocation, not just the portion addressed by a
view. Explicit host reads may allocate converted copies. Use the actual measured
backend/device capability and dtype-specific tests, not the dtype name alone, to
assess precision and performance.

## Lazy Metal storage

Metal operation results do not allocate a matching Java array until a host read needs it.
The dtype-aware `Tensor.backend().zeros(dtype, shape)` path also defers its device allocation:
it represents zeros without either backing buffer until values are consumed. A device operation
can initialize zeros on the GPU; accumulating a device gradient into a fresh zero can avoid that
initial allocation entirely. Ordinary CPU tensor constructors still allocate host storage.

Do not infer a tensor's values or validity from a missing GPU tag or a `null` `rawData()`.
Use managed operations or `get(...)`/`toFloatArray()` for values; those reads may materialize storage.
These are allocation optimizations, not changes to FLOAT32 defaults or gradient precision.

## Migrating matrix-only code

- Existing `new Tensor(rows, cols)` and `from2D(...)` helpers remain available.
- Replace `.rows` and `.cols` with `size(0)` and `size(1)`, or checked rank-two
  `matrixRows()` and `matrixCols()` in matrix-specific code.
- Replace direct `.data` access with managed `get`/`setAt` or `toFloatArray()`.
  `data()` exposes a mutable FLOAT32 backing array only for full contiguous storage;
  it is not a general ND or compact-storage accessor and bypasses mutation tracking.
- Use explicit axes and shapes instead of silently flattening tensors.

Existing FLOAT32 model and training files remain loadable without rewriting them.
New rank/dtype-aware formats preserve tensor storage and optimizer state. Precision
conversion must be explicit; do not replace real training checkpoints for experiments.

## Temporary evaluation outputs

`GradMode.noGrad()` disables gradient recording; it does not release returned
tensors. Use the library evaluator for held-out metrics. In a custom repeated
evaluation loop, consume each batch's output inside `TensorScope.open()`, close
that scope, then call `Tensor.backend().releaseTemporaryResources()`. Keep scalar
metrics, not the temporary output views. Explicitly retained model parameters
remain resident. Do not use scoped outputs afterward unless you escaped or retained
them; generic cleanup cannot safely infer that an escaped public view is no longer needed.
