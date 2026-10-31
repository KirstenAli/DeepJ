package io.github.kirstenali.deepj.tensor;

/** Parallel final-axis loss; loss-only and gradient-only modes avoid unused full-size outputs. */
abstract class GraphCrossEntropyRecordingOps extends GraphAdamWRecordingOps {
    GraphCrossEntropyRecordingOps(GpuRuntime runtime) { super(runtime); }

    public CrossEntropyResult layoutCrossEntropy(Tensor logits, int[] targets, boolean[] mask, int mode) {
        if (mode < 1 || mode > 3) throw new IllegalArgumentException("Invalid cross entropy output mode");
        long[] plan = CrossEntropyPlan.describe(logits, targets, mask);
        TensorStorage.requireSource(logits);
        requireLayoutSupport();
        GpuBuffer input = ensureStorageBuffer(logits);
        Tensor losses = mode == 2 ? null : output(DType.FLOAT32, new int[]{targets.length, 1});
        Tensor gradient = mode == 1 ? null : output(DType.FLOAT32, logits.shape());
        record(input, logits, plan, losses, gradient, mode);
        if (mask != null && gradient != null) scaleGradient(gradient, (float) plan[0] / plan[1]);
        return new CrossEntropyResult(losses, gradient);
    }

    private void scaleGradient(Tensor gradient, float scale) {
        GpuBuffer buffer = (GpuBuffer) gradient.getGpuTag();
        recordMultiplyScalar(buffer, buffer, scale);
    }

    private void record(GpuBuffer input, Tensor logits, long[] plan, Tensor losses, Tensor gradient, int mode) {
        int layout = layouts.add(logits), targets = layouts.addWords(plan);
        beginOp(7);
        emitInt(OP_LAYOUT_CROSS_ENTROPY);
        emitStorageBuffer(input);
        emitStorageBuffer((GpuBuffer) (losses == null ? gradient : losses).getGpuTag());
        emitStorageBuffer((GpuBuffer) (gradient == null ? losses : gradient).getGpuTag());
        emitInt(layout);
        emitInt(targets);
        emitInt(mode);
        endOp();
    }
}
