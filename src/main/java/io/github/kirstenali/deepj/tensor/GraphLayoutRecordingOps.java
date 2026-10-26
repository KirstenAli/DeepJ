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

    private void requireLayoutSupport() {
        if (!runtime.supportsLayouts()) {
            throw new UnsupportedOperationException("Runtime does not support GPU layout descriptors");
        }
    }

    public Tensor castToDevice(Tensor input, DType dtype) {
        TensorStorage.requireSource(input);
        Objects.requireNonNull(dtype, "dtype");
        return input.dtype() == dtype ? input : transferLayout(input, dtype);
    }
}
