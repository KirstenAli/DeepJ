package io.github.kirstenali.deepj.tensor;

/** Final-axis normalization saves FLOAT32 statistics regardless of the original storage format. */
abstract class GraphNormalizationRecordingOps extends GraphIndexRecordingOps {
    GraphNormalizationRecordingOps(GpuRuntime runtime) { super(runtime); }

    public NormalizationResult layoutNormalize(Tensor input, Tensor gamma, Tensor beta,
                                                float epsilon, boolean centered) {
        NormalizationChecks.forward(input, gamma, beta, epsilon, centered);
        Tensor[] sources = {input, gamma, centered ? beta : gamma};
        for (Tensor source : sources) { TensorStorage.requireSource(source); }
        DType dtype = NormalizationChecks.outputType(input, gamma, beta, centered);
        if (input.numel() == 0) return emptyNormalization(input, dtype);
        requireLayoutSupport();
        GpuBuffer[] values = storageBuffers(sources);
        var result = new NormalizationResult(output(dtype, input.shape()), output(DType.FLOAT32, input.shape()),
                output(DType.FLOAT32, NormalizationChecks.statisticsShape(input)));
        normalizeCommand(sources, values, result, epsilon, centered);
        return result;
    }

    private void normalizeCommand(Tensor[] sources, GpuBuffer[] values, NormalizationResult result,
                                  float epsilon, boolean centered) {
        int[] offsets = layoutOffsets(sources);
        beginOp(12);
        emitInt(OP_LAYOUT_NORMALIZE);
        for (GpuBuffer value : values) { emitStorageBuffer(value); }
        emitStorageBuffer((GpuBuffer) result.output().getGpuTag());
        emitStorageBuffer((GpuBuffer) result.normalized().getGpuTag());
        emitStorageBuffer((GpuBuffer) result.denominator().getGpuTag());
        for (int offset : offsets) { emitInt(offset); }
        emitFloatBits(epsilon);
        emitInt(centered ? 1 : 0);
        endOp();
    }

    public Tensor layoutNormalizationBackward(Tensor gradient, Tensor normalized, Tensor denominator,
                                               Tensor gamma, boolean centered) {
        NormalizationChecks.backward(gradient, normalized, denominator, gamma);
        Tensor[] sources = {gradient, normalized, denominator, gamma};
        for (Tensor source : sources) { TensorStorage.requireSource(source); }
        if (gradient.numel() == 0) return Tensor.deviceOnly(DType.FLOAT32, gradient.shape());
        requireLayoutSupport();
        GpuBuffer[] values = storageBuffers(sources);
        Tensor output = output(DType.FLOAT32, gradient.shape());
        backwardCommand(sources, values, output, centered);
        return output;
    }

    private void backwardCommand(Tensor[] sources, GpuBuffer[] values, Tensor output, boolean centered) {
        int[] offsets = layoutOffsets(sources);
        beginOp(11);
        emitInt(OP_LAYOUT_NORMALIZATION_BACKWARD);
        for (GpuBuffer value : values) { emitStorageBuffer(value); }
        emitStorageBuffer((GpuBuffer) output.getGpuTag());
        for (int offset : offsets) { emitInt(offset); }
        emitInt(centered ? 1 : 0);
        endOp();
    }

    final GpuBuffer[] storageBuffers(Tensor[] sources) {
        GpuBuffer[] values = new GpuBuffer[sources.length];
        for (int index = 0; index < values.length; index++) { values[index] = ensureStorageBuffer(sources[index]); }
        return values;
    }

    final int[] layoutOffsets(Tensor[] sources) {
        int[] offsets = new int[sources.length];
        for (int index = 0; index < offsets.length; index++) { offsets[index] = layouts.add(sources[index]); }
        return offsets;
    }

    final Tensor output(DType dtype, int[] shape) { return createOutputTensor(layoutOutput(dtype, shape), shape); }

    private static NormalizationResult emptyNormalization(Tensor input, DType dtype) {
        return new NormalizationResult(Tensor.deviceOnly(dtype, input.shape()),
                Tensor.deviceOnly(DType.FLOAT32, input.shape()),
                Tensor.deviceOnly(DType.FLOAT32, NormalizationChecks.statisticsShape(input)));
    }
}
