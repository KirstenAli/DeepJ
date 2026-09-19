package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;

public final class CrossEntropyLoss implements LossFunction {

    private static final float TARGET_INTEGER_EPS = 1e-6f;

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
        return logits.crossEntropyGradient(targets);
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
        if (actual.cols != 1) {
            throw new IllegalArgumentException(
                    "CrossEntropyLoss expects targets shape [n x 1], got [" + actual.rows + " x " + actual.cols + "]"
            );
        }

        actual.materialize();
        int[] y = new int[actual.rows];
        for (int i = 0; i < actual.rows; i++) y[i] = requireIntegerTarget(actual.data[i], i);
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
        if (logits == null) {
            throw new IllegalArgumentException("logits is null");
        }
        if (targets == null) {
            throw new IllegalArgumentException("targets is null");
        }
        Tensor.requireTargetsMatchRows(logits, targets);
        for (int t : targets) {
            if (t < 0 || t >= logits.cols) {
                throw new IllegalArgumentException(
                        "target id out of range: " + t + " (vocab=" + logits.cols + ")"
                );
            }
        }
    }

    private static void checkMask(Tensor logits, boolean[] mask) {
        if (mask == null || mask.length != logits.rows) {
            throw new IllegalArgumentException("loss mask must match logits rows");
        }
        for (boolean included : mask) if (included) return;
        throw new IllegalArgumentException("loss mask must include at least one row");
    }
}
