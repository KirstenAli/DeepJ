# Tensors, shapes and precision

DeepJ uses one dense `Tensor` API for scalars, vectors, matrices and higher-rank
arrays. Existing matrices are rank-two tensors; values default to FLOAT32.

These ND and optional-dtype APIs currently require a source build; they are not
part of the published `0.11.0-alpha` artifact.

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
accumulation. Eligible larger FLOAT16 products use Apple MPS on supported Apple GPUs;
smaller, strided and other dtype/device combinations retain their appropriate GPU
paths. Selection preserves mixed-input precision without full-matrix conversion copies.
Other operations and unsupported native compact-matrix paths may use
FLOAT32 arithmetic on the selected backend. This is not automatic mixed precision:
there is no implicit autocast or loss scaling, and no hidden CPU fallback from Metal.
Optimizer master weights, gradients and moments remain FLOAT32. Consequently, two-byte
weights do not halve total training memory or guarantee faster training.

`MetalBackend.supportsNative16Matmul(dtype)` reports native compact-matrix support
on the active device. It does not promise that every operation uses compact arithmetic.

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
