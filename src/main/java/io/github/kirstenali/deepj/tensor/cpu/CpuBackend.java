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

    @Override public NormalizationResult normalize(Tensor input, Tensor gamma, Tensor beta,
                                                    float epsilon, boolean centered) {
        return CpuNdNormalization.apply(input, gamma, beta, epsilon, centered);
    }

    @Override public Tensor normalizationBackward(Tensor gradient, Tensor normalized, Tensor denominator,
                                                    Tensor gamma, boolean centered) {
        return CpuNdNormalization.backward(gradient, normalized, denominator, gamma, centered);
    }

    @Override public RmsNormResult rmsNorm(Tensor input, Tensor gamma, float epsilon) {
        if (CpuNdElementwise.matrix(input) && CpuNdElementwise.matrix(gamma)) {
            return super.rmsNorm(input, gamma, epsilon);
        }
        var result = normalize(input, gamma, null, epsilon, false);
        return new RmsNormResult(result.output(), result.normalized(), result.denominator());
    }

    @Override public Tensor rmsNormBackward(Tensor gradient, Tensor normalized, Tensor rms, Tensor gamma) {
        if (CpuNdElementwise.sameMatrices(gradient, normalized) && CpuNdElementwise.matrix(gamma)
                && CpuNdElementwise.matrix(rms)) return super.rmsNormBackward(gradient, normalized, rms, gamma);
        return normalizationBackward(gradient, normalized, rms, gamma, false);
    }

    @Override public Tensor swiGlu(Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        return CpuNdElementwise.sameMatrices(gate, up) ? super.swiGlu(gate, up) : CpuNdSwiGlu.apply(gate, up);
    }

    @Override public SwiGluBackwardResult swiGluBackward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        Tensor.requireSameShape(gate, gradient, "SwiGLU gradient");
        return CpuNdElementwise.sameMatrices(gate, up) && CpuNdElementwise.sameMatrices(gate, gradient)
                ? super.swiGluBackward(gradient, gate, up) : CpuNdSwiGlu.backward(gradient, gate, up);
    }
}
