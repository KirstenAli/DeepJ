package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.*;

public final class CpuBackend extends CpuTrainingOps {
    @Override public Tensor add(Tensor left, Tensor right) { return CpuAddition.apply(left, right); }
    @Override public void addInPlace(Tensor left, Tensor right) { CpuAddition.into(left, right, left); }

    @Override public Tensor causalSoftmax(Tensor input, int length, float scale) {
        return CpuNdElementwise.matrix(input) ? CpuCausalSoftmax.apply(input, length, scale)
                : CpuNdPosition.softmax(input, length, scale);
    }

    @Override public Tensor sliceRows(Tensor input, int[] rows) {
        MatrixShapeChecks.requireMatrix(input);
        for (int row : rows) {
            if (row < 0 || row >= input.matrixRows()) throw new IllegalArgumentException("Row index out of range");
        }
        return CpuNdIndexing.select(input, 0, rows);
    }

    @Override public Tensor splitHeads(Tensor input, int heads) {
        return CpuNdElementwise.matrix(input) ? super.splitHeads(input, heads)
                : PackedLayouts.split(this, input, heads);
    }

    @Override public Tensor mergeHeads(Tensor input, int heads) {
        return CpuNdElementwise.matrix(input) ? super.mergeHeads(input, heads)
                : PackedLayouts.merge(this, input, heads);
    }

    @Override public Tensor matmul(Tensor left, Tensor right) {
        Tensor result = CpuMatmul.attempt(left, right);
        return result == null ? super.matmul(left, right) : result;
    }

    @Override public Tensor batchedMatmul(Tensor left, Tensor right, int batches, boolean ta, boolean tb) {
        Tensor result = CpuPackedProduct.attempt(left, right, batches, ta, tb);
        if (result != null) return result;
        return CpuNdElementwise.matrix(left) && CpuNdElementwise.matrix(right)
                ? super.batchedMatmul(left, right, batches, ta, tb)
                : PackedLayouts.matmul(this, left, right, batches, ta, tb);
    }

    @Override public Tensor headPrefixMatmul(Tensor left, Tensor cache, int heads, int length, boolean transpose) {
        return CpuHeadPrefixProduct.apply(this, left, cache, heads, length, transpose);
    }

    @Override public void copyInto(Tensor target, Tensor source) { CpuNdAssignment.apply(target, source); }
    @Override public Tensor zeros(DType dtype, int... shape) { return CpuMatrixStorage.zeros(dtype, shape); }

    @Override public Tensor zerosForLoading(DType dtype, int... shape) {
        return CpuMatrixStorage.zerosForLoading(dtype, shape);
    }

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

    @Override public Tensor softmaxAxis(Tensor input, int axis) {
        if (!matrixAxis(input, axis)) return CpuNdSoftmax.apply(input, axis);
        return reshape(softmaxRows(reshape(input, -1, input.size(-1))), input.shape());
    }

    @Override public Tensor softmaxAxisBackward(Tensor gradient, Tensor output, int axis) {
        if (!matrixAxis(output, axis) || !matrixAxis(gradient, axis) || !output.hasSameShape(gradient)) {
            return super.softmaxAxisBackward(gradient, output, axis);
        }
        Tensor result = softmaxBackward(reshape(gradient, -1, output.size(-1)),
                reshape(output, -1, output.size(-1)));
        return reshape(result, output.shape());
    }

    private static boolean matrixAxis(Tensor input, int axis) {
        return input.rank() > 0 && input.numel() > 0 && (axis == -1 || axis == input.rank() - 1)
                && input.dtype() == DType.FLOAT32 && input.coversStorage();
    }

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
        return CpuNdElementwise.matrix(logits) ? CpuMatrixCrossEntropy.apply(logits, targets, null)
                : CpuNdCrossEntropy.apply(logits, targets, null);
    }

    @Override public CrossEntropyResult crossEntropy(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.included(mask, CrossEntropyChecks.positions(logits));
        return CpuNdElementwise.matrix(logits) ? CpuMatrixCrossEntropy.apply(logits, targets, mask)
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
        var result = normalize(input, gamma, null, epsilon, false);
        return new RmsNormResult(result.output(), result.normalized(), result.denominator());
    }

    @Override public Tensor rmsNormBackward(Tensor gradient, Tensor normalized, Tensor rms, Tensor gamma) {
        return normalizationBackward(gradient, normalized, rms, gamma, false);
    }

    @Override public Tensor swiGlu(Tensor gate, Tensor up) {
        return CpuNdSwiGlu.apply(gate, up);
    }

    @Override public SwiGluBackwardResult swiGluBackward(Tensor gradient, Tensor gate, Tensor up) {
        return CpuNdSwiGlu.backward(gradient, gate, up);
    }
}
