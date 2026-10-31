package io.github.kirstenali.deepj.tensor;

/** Maximum groups count ties once and scatter their FLOAT32 adjoint to the original logical layout. */
abstract class GraphMaximumRecordingOps extends GraphReductionRecordingOps {
    GraphMaximumRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutMaximumGradient(Tensor input, Tensor maximum, Tensor gradient, int[] axes, boolean keep) {
        ReductionShape shape = requireMaximumShape(input, maximum, gradient, axes, keep);
        if (input.numel() == 0) return Tensor.deviceOnly(DType.FLOAT32, input.shape());
        requireLayoutSupport();
        return maximumToDevice(input, maximum, castToDevice(gradient, DType.FLOAT32), shape);
    }

    private static ReductionShape requireMaximumShape(Tensor input, Tensor maximum, Tensor gradient,
                                                       int[] axes, boolean keep) {
        TensorStorage.requireSource(input);
        Tensor.requireSameShape(maximum, gradient, "Maximum gradient");
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        if (!maximum.layout().shape().equals(shape.output())) {
            throw new IllegalArgumentException("Invalid maximum shape");
        }
        if (maximum.dtype() != input.dtype()) {
            throw new IllegalArgumentException("Maximum adjoint requires the original saved dtype");
        }
        return shape;
    }

    private Tensor maximumToDevice(Tensor input, Tensor maximum, Tensor gradient, ReductionShape shape) {
        ensureStorageBuffer(input);
        ensureStorageBuffer(maximum);
        ensureStorageBuffer(gradient);
        Tensor output = createOutputTensor(layoutOutput(DType.FLOAT32, input.shape()), input.shape());
        maximumCommand(TensorViews.permute(input, shape.groupedAxes()), maximum, gradient,
                TensorViews.permute(output, shape.groupedAxes()), shape.retainedRank());
        return output;
    }

    private void maximumCommand(Tensor input, Tensor maximum, Tensor gradient, Tensor output, int retained) {
        GpuBuffer[] buffers = {ensureStorageBuffer(input), ensureStorageBuffer(maximum),
                ensureStorageBuffer(gradient), ensureStorageBuffer(output)};
        int[] offsets = {layouts.add(input), layouts.add(maximum), layouts.add(gradient), layouts.add(output)};
        beginOp(10);
        emitInt(OP_LAYOUT_MAXIMUM_BACKWARD);
        for (GpuBuffer buffer : buffers) { emitStorageBuffer(buffer); }
        for (int offset : offsets) { emitInt(offset); }
        emitInt(retained);
        endOp();
    }
}
