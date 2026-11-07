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
FLOAT16 and BFLOAT16 matrix-vector products use CPU SIMD kernels with FLOAT32 accumulation;
eligible FP16 matrix-matrix products use BNNS with compact operands and bounded workspace.
Other compact matrix products convert bounded tiles for BLAS. Small products retain their existing native or Java routes. This CPU library loads independently of Metal and
falls back to Java if unavailable. `-Ddeepj.cpu.javaOnly=true` disables it for a JVM.

Batched FLOAT32 products pin their arrays once per batch operation. Compatible
transposed inputs also use BLAS, including the products used for weight gradients.
Dense FLOAT32 and contiguous compact normalization use native forward and backward
loops with the same scalar reduction order as Java. Compact statistics and gradients
remain FLOAT32. Compact addition, SwiGLU and softmax avoid scalar Java dtype conversion;
softmax retries individual numerically sensitive rows through Java when needed.
FLOAT32 causal softmax uses native vector exponentials for
4,096–262,144 elements, distributes independent rows within the configured thread
limit, and retains the parallel Java path for larger matrices.
Unsupported storage and values retain their Java fallback.

Dense FLOAT32 rotary positions share an array-based CPU implementation for
interleaved and half-split pairs, including backward. Large inputs use the CPU
executor; small inputs execute directly. Contiguous compact rotary positions use the
same pairing rules with direct compact storage access; strided inputs keep their fallback.

Large compact matrices allocated through `CpuBackend.zeros(...)` can use column-major
storage for contiguous dot products. Checkpoint slots allocated through
`ParameterFactory.forLoading()` also use this layout for large FLOAT32 projections.
Tall lookup tables retain row-major storage. Ordinary FLOAT32 zero tensors, gradient
buffers and randomly initialized parameters retain their contiguous layout.
Metal checkpoint slots also use column-major storage for eligible large compact
projections; its FLOAT32 loading layout remains unchanged.
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
results and precision policies. The [full Qwen3 follow-up](benchmarks/full-qwen-2026-11-07/REPORT.md)
records the current FP32 CPU/Metal comparison and training regression controls.

## Shared network operations

- Dense layers and MLPs use the existing projections and matrix products.
- Convolution uses native Metal execution for supported inference shapes and
  differentiable window gathering and matrix multiplication for training and fallback.
- Pooling and BatchNorm use tensor reductions.
- Recurrent cells use linear projections, tensor gates, and activations.
- Transformer encoders use batched products, softmax, normalization, and residual operations.

These operations dispatch through the selected backend. Metadata indices, sequence
masks, attention masks, and dropout masks are constructed on the host; network
activations are not read into Java arrays to calculate layer outputs or gradients.

Metal inference supports same-padded NCHW grouped and depthwise convolution through
MPSGraph in FP32, FP16 and supported BF16 configurations. Gradient-enabled execution
keeps the differentiable composition, whose gathered-window memory depends on batch
size, output positions, channels, and kernel size. Transformer-specific fused attention support has separate shape
and device requirements. See [tensors and precision](tensors.md) for the detailed
Metal matrix and attention paths.

## Precision and host transfers

FLOAT32 is the default. FLOAT16 and BFLOAT16 provide optional compact storage with
explicit promotion and FLOAT32 gradients/optimizer state. Storage width does not
promise native compact arithmetic for every operation, faster training, or a fixed
total-memory reduction. See [precision](tensors.md#storage-is-not-the-same-as-calculation-precision).

Compact cached decoding can combine score products, scaling, softmax and value
products in one Metal dispatch. This shared inference path supports FP16/BF16
head widths up to 128 and valid cache prefixes up to 512 positions, including
grouped-query heads. Other shapes and FP32 use the existing cache operations.
Gradient-enabled execution never selects this inference kernel.

Captured model inference reuses scratch within bounded blocks and keeps block
outputs on the device. Channel RMS normalization preserves channel-first storage;
compact diffusion normalization rounds its affine operations in the parameter dtype.

Eligible compact MPSGraph products use the GPU compilation setting and submit
ordered batches while the host prepares later work. Every submitted batch is
finished and checked before a flush returns, including failure cleanup. See the
[compact execution measurements](benchmarks/compact-followup-2026-11-07/REPORT.md).

Explicit value reads such as `get(...)`, `toFloatArray()`, and `item()` can materialize
host data. Loss reporting and persistence use such transfers where needed. Keep model
arithmetic in tensor operations when gradients or device execution are required.

## Evaluation and temporary tensors

`GradMode.noGrad()` disables graph recording. It does not change a module's
training/evaluation mode or define the lifetime of returned tensors. Consume temporary
evaluation outputs inside `TensorScope`, close the scope, then call
`Tensor.backend().releaseTemporaryResources()` at an appropriate cleanup boundary.
Escape or explicitly retain a tensor that must outlive its scope.

Metal keeps a bounded cache of released native scratch buffers: up to 128 buffers,
32 MiB total, and 4 MiB per buffer. Reuse follows command completion and requires an
exact byte-size match. The cache empties when the last native buffer is released;
configured `deepj.metal.maxBufferBytes` budgets also clear it after releases and
allocation rollbacks. Set `DEEPJ_METAL_DISABLE_BUFFER_REUSE=1` before launching the JVM
to disable it for diagnosis.

See [shared-operation performance measurements](benchmarks/shared-tensor-2026-11-06/REPORT.md)
for the matched inference and training probes.

See [autograd](autograd.md) for graph and mutation rules and
[training](training.md#training-and-evaluation-mode) for dropout and BatchNorm behavior.

## Verification

The portable build runs CPU and backend-independent checks. The full macOS profile
requires an actual Metal device and desktop session and checks device numerics and
coverage. A skipped device test is not a Metal pass. See [tests and coverage](testing.md)
and [regression standards](quality-and-regression.md) for the commands and scope.
