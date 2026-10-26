package io.github.kirstenali.deepj.tensor;

/** Adam operates on FLOAT32 parameters or masters and moments, including independent strided views. */
abstract class GraphAdamWRecordingOps extends GraphSwiGluRecordingOps {
    GraphAdamWRecordingOps(GpuRuntime runtime) { super(runtime); }

    public void layoutAdamW(Tensor weight, Tensor gradient, Tensor first, Tensor second, float lr,
                            float beta1, float beta2, float epsilon, float decay, float bc1, float bc2) {
        OptimizerChecks.requireState(weight, gradient, first, second);
        Tensor[] tensors = {weight, gradient, first, second};
        for (Tensor tensor : tensors) { TensorStorage.requireSource(tensor); }
        if (weight.numel() == 0) return;
        requireLayoutSupport();
        GpuBuffer[] buffers = storageBuffers(tensors);
        int[] offsets = layoutOffsets(tensors);
        command(buffers, offsets, new float[]{lr, beta1, beta2, epsilon, decay, bc1, bc2});
        markUpdated(weight);
        markUpdated(first);
        markUpdated(second);
    }

    private void command(GpuBuffer[] buffers, int[] offsets, float[] parameters) {
        beginOp(16);
        emitInt(OP_LAYOUT_ADAMW);
        for (GpuBuffer buffer : buffers) { emitStorageBuffer(buffer); }
        for (int offset : offsets) { emitInt(offset); }
        for (float parameter : parameters) { emitFloatBits(parameter); }
        endOp();
    }
}
