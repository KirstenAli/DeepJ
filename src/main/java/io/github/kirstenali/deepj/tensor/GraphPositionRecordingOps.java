package io.github.kirstenali.deepj.tensor;

/** Position metadata describes logical matrices independently of their physical typed allocations. */
abstract class GraphPositionRecordingOps extends GraphCrossEntropyRecordingOps {
    GraphPositionRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutCausal(Tensor input, int length, float scale, int mode) {
        MatrixShapeChecks.requireCausal(input, length);
        CpuNdPosition.requireScale(scale);
        if (mode != 1 && mode != 2) throw new IllegalArgumentException("Invalid causal operation mode");
        TensorStorage.requireSource(input);
        if (input.numel() == 0) return Tensor.deviceOnly(input.dtype(), input.shape());
        requireLayoutSupport();
        GpuBuffer source = ensureStorageBuffer(input);
        Tensor result = output(input.dtype(), input.shape());
        causalCommand(source, input, result, length, scale, mode);
        return result;
    }

    private void causalCommand(GpuBuffer source, Tensor input, Tensor result, int length, float scale, int mode) {
        int layout = layouts.add(input);
        beginOp(7);
        emitInt(OP_LAYOUT_CAUSAL);
        emitStorageBuffer(source);
        emitStorageBuffer((GpuBuffer) result.getGpuTag());
        emitInt(layout);
        emitInt(length);
        emitFloatBits(scale);
        emitInt(mode);
        endOp();
    }

    public Tensor layoutRotary(Tensor input, Tensor cosine, Tensor sine, int length, boolean inverse) {
        return layoutRotary(input, cosine, sine, length, inverse, false);
    }

    public Tensor layoutRotary(Tensor input, Tensor cosine, Tensor sine, int length,
                                boolean inverse, boolean halfSplit) {
        MatrixShapeChecks.requireRotary(input, cosine, sine, length);
        Tensor[] sources = {input, cosine, sine};
        for (Tensor source : sources) { TensorStorage.requireSource(source); }
        if (input.numel() == 0) return Tensor.deviceOnly(input.dtype(), input.shape());
        requireLayoutSupport();
        GpuBuffer[] values = storageBuffers(sources);
        Tensor result = output(input.dtype(), input.shape());
        rotaryCommand(sources, values, result, length, (inverse ? 1 : 0) | (halfSplit ? 2 : 0));
        return result;
    }

    private void rotaryCommand(Tensor[] sources, GpuBuffer[] values, Tensor result, int length, int flags) {
        int[] offsets = layoutOffsets(sources);
        beginOp(10);
        emitInt(OP_LAYOUT_ROTARY);
        for (GpuBuffer value : values) { emitStorageBuffer(value); }
        emitStorageBuffer((GpuBuffer) result.getGpuTag());
        for (int offset : offsets) { emitInt(offset); }
        emitInt(length);
        emitInt(flags);
        endOp();
    }
}
