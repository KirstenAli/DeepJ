package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.CrossEntropyChecks;
import io.github.kirstenali.deepj.autograd.Autograd;

public final class CrossEntropyLoss implements LossFunction {

    private static final float TARGET_INTEGER_EPS = 1e-6f;

    @Override
    public Tensor forward(Tensor predicted, Tensor actual) {
        return forward(predicted, toIntTargets(actual), null);
    }

    public static Tensor forward(Tensor logits, int[] targets, boolean[] mask) {
        checkTargets(logits, targets);
        CrossEntropyResult result = mask == null ? result(logits, targets) : maskedResult(logits, targets, mask);
        Tensor gradient = result.gradient();
        return Autograd.recordSaved(result.loss(), g -> new Tensor[]{scaleGradient(gradient, g)},
                new Tensor[]{logits}, gradient);
    }

    private static CrossEntropyResult maskedResult(Tensor logits, int[] targets, boolean[] mask) {
        checkMask(logits, mask);
        return Tensor.backend().crossEntropy(logits, targets, mask);
    }

    private static Tensor scaleGradient(Tensor gradient, Tensor scalar) {
        if (scalar.getGpuTag() == null) {
            float scale = scalar.item();
            return scale == 1.0f ? gradient : gradient.multiplyScalar(scale);
        }
        if (gradient.rank() != 2) return gradient.multiply(scalar);
        Tensor weights = Tensor.ones(gradient.matrixRows(), 1).multiplyBroadcastRows(scalar);
        return gradient.multiplyBroadcastCols(weights);
    }

    @Override
    public float loss(Tensor predicted, Tensor actual) {
        int[] y = toIntTargets(actual);
        return loss(predicted, y);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        int[] y = toIntTargets(actual);
        return gradient(predicted, y);
    }

    public static float loss(Tensor logits, int[] targets) {
        checkTargets(logits, targets);
        return logits.crossEntropyLoss(targets);
    }

    public static Tensor gradient(Tensor logits, int[] targets) {
        checkTargets(logits, targets);
        return Tensor.backend().crossEntropyGradient(logits, targets);
    }

    public static CrossEntropyResult result(Tensor logits, int[] targets) {
        checkTargets(logits, targets);
        return Tensor.backend().crossEntropy(logits, targets);
    }

    public static float loss(Tensor logits, int[] targets, boolean[] mask) {
        checkTargets(logits, targets);
        checkMask(logits, mask);
        return Tensor.backend().crossEntropyLoss(logits, targets, mask);
    }

    public static Tensor gradient(Tensor logits, int[] targets, boolean[] mask) {
        checkTargets(logits, targets);
        checkMask(logits, mask);
        return Tensor.backend().crossEntropyGradient(logits, targets, mask);
    }

    public static int[] toIntTargets(Tensor actual) {
        float[] values = actual.toFloatArray();
        int[] y = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            y[i] = requireIntegerTarget(values[i], i);
        }
        return y;
    }

    private static int requireIntegerTarget(float value, int row) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException("target value at row " + row + " must be finite");
        }
        int asInt = (int) value;
        if (Math.abs(value - asInt) > TARGET_INTEGER_EPS) {
            throw new IllegalArgumentException(
                    "target value at row " + row + " must be an integer class id, got " + value);
        }
        return asInt;
    }

    public static Tensor fromIntTargets(int[] targets) {
        return TensorAdapters.fromIntColumn(targets);
    }

    private static void checkTargets(Tensor logits, int[] targets) {
        CrossEntropyChecks.requireTargets(logits, targets);
    }

    private static void checkMask(Tensor logits, boolean[] mask) {
        CrossEntropyChecks.included(mask, CrossEntropyChecks.positions(logits));
    }
}
