package io.github.kirstenali.deepj.tensor;

/** Typed saved activations decode directly on GPU; both SwiGLU adjoints are FLOAT32. */
abstract class GraphSwiGluRecordingOps extends GraphNormalizationRecordingOps {
    GraphSwiGluRecordingOps(GpuRuntime runtime) { super(runtime); }

    public SwiGluBackwardResult layoutSwiGluBackward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        Tensor.requireSameShape(gate, gradient, "SwiGLU gradient");
        Tensor[] inputs = {gradient, gate, up};
        for (Tensor input : inputs) { TensorStorage.requireSource(input); }
        if (gate.numel() == 0) return new SwiGluBackwardResult(Tensor.deviceOnly(DType.FLOAT32, gate.shape()),
                Tensor.deviceOnly(DType.FLOAT32, gate.shape()));
        requireLayoutSupport();
        GpuBuffer[] sources = storageBuffers(inputs);
        Tensor first = output(DType.FLOAT32, gate.shape()), second = output(DType.FLOAT32, up.shape());
        command(sources, layoutOffsets(inputs), first, second);
        return new SwiGluBackwardResult(first, second);
    }

    private void command(GpuBuffer[] inputs, int[] offsets, Tensor first, Tensor second) {
        beginOp(9);
        emitInt(OP_LAYOUT_SWIGLU_BACKWARD);
        for (GpuBuffer input : inputs) { emitStorageBuffer(input); }
        emitStorageBuffer((GpuBuffer) first.getGpuTag());
        emitStorageBuffer((GpuBuffer) second.getGpuTag());
        for (int offset : offsets) { emitInt(offset); }
        endOp();
    }
}
