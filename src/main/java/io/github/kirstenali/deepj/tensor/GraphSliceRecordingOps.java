package io.github.kirstenali.deepj.tensor;

/** Slice adjoints initialize one dense result, then assign only the selected logical region on GPU. */
abstract class GraphSliceRecordingOps extends GraphMaximumRecordingOps {
    GraphSliceRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutSliceGradient(Tensor gradient, int[] dimensions, int axis, int start, int step) {
        TensorLayout region = SliceGradientLayout.of(gradient, dimensions, axis, start, step);
        if (gradient.numel() != 0) requireLayoutSupport();
        Tensor source = castToDevice(gradient, DType.FLOAT32);
        if (gradient.numel() != 0) ensureStorageBuffer(source);
        Tensor output = zeroToDevice(DType.FLOAT32, dimensions);
        assignToDevice(Tensor.aliasStorage(output, region), source);
        return output;
    }
}
