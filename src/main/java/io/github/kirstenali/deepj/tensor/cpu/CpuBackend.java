package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.*;

public final class CpuBackend extends CpuTrainingOps {
    @Override public Tensor zeros(DType dtype, int... shape) { return new Tensor(dtype, shape); }

    @Override public Tensor cast(Tensor input, DType dtype) { return CpuNdElementwise.cast(input, dtype); }

    @Override public Tensor sumAxes(Tensor input, int[] axes, boolean keepDimensions) {
        return CpuNdReduction.sum(input, axes, keepDimensions);
    }

    @Override public Tensor meanAxes(Tensor input, int[] axes, boolean keepDimensions) {
        return CpuNdReduction.mean(input, axes, keepDimensions);
    }

    @Override public Tensor varianceAxes(Tensor input, int[] axes, boolean keep, int correction) {
        return CpuNdStatistics.variance(input, axes, keep, correction);
    }

    @Override public Tensor maximumAxes(Tensor input, int[] axes, boolean keep) {
        return CpuNdStatistics.maximum(input, axes, keep);
    }

    @Override public Tensor maximumAxesBackward(Tensor input, Tensor maximum, Tensor gradient,
                                                int[] axes, boolean keep) {
        return CpuNdStatistics.maximumGradient(input, maximum, gradient, axes, keep);
    }

    @Override public Tensor sliceGradient(Tensor gradient, int[] shape, int axis, int start, int step) {
        return CpuNdReduction.sliceGradient(gradient, shape, axis, start, step);
    }

    @Override public Tensor softmaxAxis(Tensor input, int axis) { return CpuNdSoftmax.apply(input, axis); }

    @Override public Tensor indexSelect(Tensor input, int axis, int[] indices) {
        return CpuNdIndexing.select(input, axis, indices);
    }

    @Override public void indexAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        CpuNdIndexing.add(target, axis, indices, updates);
    }

    @Override public Tensor indexSelectGradient(Tensor gradient, int[] shape, int axis, int[] indices) {
        return CpuNdIndexing.gradient(gradient, shape, axis, indices);
    }

    @Override public CrossEntropyResult crossEntropy(Tensor logits, int[] targets) {
        return CpuNdElementwise.matrix(logits) ? CrossEntropyDefaults.apply(this, logits, targets, null)
                : CpuNdCrossEntropy.apply(logits, targets, null);
    }

    @Override public CrossEntropyResult crossEntropy(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.included(mask, CrossEntropyChecks.positions(logits));
        return CpuNdElementwise.matrix(logits) ? CrossEntropyDefaults.apply(this, logits, targets, mask)
                : CpuNdCrossEntropy.apply(logits, targets, mask);
    }
}
