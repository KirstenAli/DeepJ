package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Layout-aware commands retain fixed-width buffer arguments for lifetime analysis. */
abstract class GraphLayoutRecordingOps extends GraphCore {
    GraphLayoutRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor transferLayout(Tensor input, DType dtype) {
        TensorStorage.requireSource(input);
        Objects.requireNonNull(dtype, "dtype");
        if (input.numel() == 0) return Tensor.deviceOnly(dtype, input.shape());
        requireLayoutSupport();
        return recordTransfer(input, dtype);
    }

    private Tensor recordTransfer(Tensor input, DType dtype) {
        GpuBuffer source = ensureStorageBuffer(input);
        int rows = input.rank() == 2 ? input.matrixRows() : 1;
        GpuBuffer output = newOutputBuffer(rows, input.numel() / rows, dtype);
        int layout = layouts.add(input);
        beginOp(4);
        emitInt(OP_LAYOUT_TRANSFER);
        emitStorageBuffer(source);
        emitStorageBuffer(output);
        emitInt(layout);
        endOp();
        return createOutputTensor(output, input.shape());
    }

    final void requireLayoutSupport() {
        if (!runtime.supportsLayouts()) {
            throw new UnsupportedOperationException("Runtime does not support GPU layout descriptors");
        }
    }

    final GpuBuffer layoutOutput(DType dtype, int... shape) {
        int count = new TensorShape(shape).elements();
        int rows = shape.length == 2 ? shape[0] : 1;
        return newOutputBuffer(rows, count / rows, dtype);
    }

    public Tensor zeroToDevice(DType dtype, int... shape) {
        Objects.requireNonNull(dtype, "dtype");
        int count = new TensorShape(shape).elements();
        if (count == 0) return Tensor.deviceOnly(dtype, shape);
        int rows = shape.length == 2 ? shape[0] : 1;
        return createOutputTensor(newZeroBuffer(rows, count / rows, dtype), shape);
    }

    /** Zeros that stay unallocated until first use; a device zero-fill is queued only when a kernel consumes them. */
    public Tensor lazyZeros(DType dtype, int... shape) {
        Objects.requireNonNull(dtype, "dtype");
        new TensorShape(shape);
        return Tensor.deviceOnly(dtype, shape);
    }

    public Tensor castToDevice(Tensor input, DType dtype) {
        TensorStorage.requireSource(input);
        Objects.requireNonNull(dtype, "dtype");
        return input.dtype() == dtype ? input : transferLayout(input, dtype);
    }
}
