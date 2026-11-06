# CPU and Metal execution

DeepJ's neural networks share one tensor and autograd system. The selected tensor
backend performs their numerical operations, including forward calculations and
first-order gradients.

## Backend selection

The default backend is `CpuBackend`, with portable Java kernels and optional native
CPU matrix acceleration on Apple Silicon running macOS 15 or newer.
`Tensor.setBackend(...)` selects a backend globally. For explicit Metal execution, check
`MetalBackend.isAvailable()` and select a `MetalBackend` before constructing the
model and tensors. Explicit Metal construction fails when a device is unavailable.

Metal acceleration requires supported Apple Silicon macOS. The CPU backend supports
other Java platforms. The optional language-model chat tools have their own automatic
backend selection; constructing a generic network does not select Metal automatically.

Backend selection is part of tensor ownership. Do not switch backends between forward
and backward or while GPU-backed model state is in use. A backward graph verifies
that it is executed on the backend that recorded it.

## CPU matrix execution

On supported Apple Silicon Macs, large FLOAT32 products use Accelerate BLAS.
FLOAT16 and BFLOAT16 weights use CPU SIMD kernels with FLOAT32 accumulation;
multi-position products convert bounded tiles for BLAS. Small and unsupported
products retain Java execution. This CPU library loads independently of Metal and
falls back to Java if unavailable. `-Ddeepj.cpu.javaOnly=true` disables it for a JVM.

Large compact matrices allocated through `CpuBackend.zeros(...)` can use column-major
storage for contiguous dot products. Checkpoint slots allocated through
`ParameterFactory.forLoading()` also use this layout for large FLOAT32 projections.
Tall lookup tables retain row-major storage. Ordinary FLOAT32 zero tensors, gradient
buffers and randomly initialized parameters retain their contiguous layout.
Logical shapes, values, dtype promotion, gradients and checkpoint order stay the same;
use tensor accessors rather than assuming a physical layout. `Tensor.zeros(...)`
continues to create contiguous tensors. No expanded weight cache is retained.

Compact tiles use `DeepJExecutor`; Accelerate manages its own CPU threads. Disabling
DeepJ parallel execution, or selecting one thread, selects serial BLAS calls.
`VECLIB_MAXIMUM_THREADS` can cap Accelerate threads before starting the JVM.
Single CPU work items execute locally, avoiding an unnecessary worker handoff.

The packaged CPU library can be rebuilt independently of Metal:

```bash
JAVA_HOME="$(/usr/libexec/java_home -v 20)" bash native/build-cpu-macos.sh
```

See [CPU performance measurements](cpu-performance.md) for the benchmark setup,
results and precision policies.

## Shared network operations

- Dense layers and MLPs use the existing projections and matrix products.
- Convolution uses gathered windows and matrix multiplication.
- Pooling and BatchNorm use tensor reductions.
- Recurrent cells use linear projections, tensor gates, and activations.
- Transformer encoders use batched products, softmax, normalization, and residual operations.

These operations dispatch through the selected backend. Metadata indices, sequence
masks, attention masks, and dropout masks are constructed on the host; network
activations are not read into Java arrays to calculate layer outputs or gradients.

Convolution materializes gathered windows, so its temporary memory depends on batch
size, output positions, channels, and kernel size. It does not use a new fused native
convolution kernel. Transformer-specific fused attention support has separate shape
and device requirements. See [tensors and precision](tensors.md) for the detailed
Metal matrix and attention paths.

## Precision and host transfers

FLOAT32 is the default. FLOAT16 and BFLOAT16 provide optional compact storage with
explicit promotion and FLOAT32 gradients/optimizer state. Storage width does not
promise native compact arithmetic for every operation, faster training, or a fixed
total-memory reduction. See [precision](tensors.md#storage-is-not-the-same-as-calculation-precision).

Explicit value reads such as `get(...)`, `toFloatArray()`, and `item()` can materialize
host data. Loss reporting and persistence use such transfers where needed. Keep model
arithmetic in tensor operations when gradients or device execution are required.

## Evaluation and temporary tensors

`GradMode.noGrad()` disables graph recording. It does not change a module's
training/evaluation mode or define the lifetime of returned tensors. Consume temporary
evaluation outputs inside `TensorScope`, close the scope, then call
`Tensor.backend().releaseTemporaryResources()` at an appropriate cleanup boundary.
Escape or explicitly retain a tensor that must outlive its scope.

See [autograd](autograd.md) for graph and mutation rules and
[training](training.md#training-and-evaluation-mode) for dropout and BatchNorm behavior.

## Verification

The portable build runs CPU and backend-independent checks. The full macOS profile
requires an actual Metal device and desktop session and checks device numerics and
coverage. A skipped device test is not a Metal pass. See [tests and coverage](testing.md)
and [regression standards](quality-and-regression.md) for the commands and scope.
