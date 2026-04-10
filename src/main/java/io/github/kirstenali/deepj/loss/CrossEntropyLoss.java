package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Cross-entropy loss with integer class targets.
 *
 * <p>Expected shapes:
 * <ul>
 *   <li>predicted (logits): [nTokens x vocab]</li>
 *   <li>actual (class indices): [nTokens x 1] where each entry is an integer in [0, vocab)</li>
 * </ul>
 *
 * <p>This class also provides helpers for common language-modeling usage where targets are provided as int[].
 */
public final class CrossEntropyLoss implements LossFunction {

    @Override
    public double loss(Tensor predicted, Tensor actual) {
        int[] y = toIntTargets(actual);
        return loss(predicted, y);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        int[] y = toIntTargets(actual);
        return gradient(predicted, y);
    }

    /**
     * Convenience helper: compute loss from logits and int targets.
     */
    public static double loss(Tensor logits, int[] targets) {
        checkTargets(logits, targets);
        return logits.crossEntropyLoss(targets);
    }

    /**
     * Convenience helper: gradient w.r.t. logits, averaged over rows.
     */
    public static Tensor gradient(Tensor logits, int[] targets) {
        checkTargets(logits, targets);
        return logits.crossEntropyGradient(targets);
    }

    /**
     * Converts a [n x 1] Tensor of class indices into an int[].
     */
    public static int[] toIntTargets(Tensor actual) {
        if (actual.cols != 1) {
            throw new IllegalArgumentException(
                    "CrossEntropyLoss expects targets shape [n x 1], got [" + actual.rows + " x " + actual.cols + "]"
            );
        }

        actual.materialize();
        int[] y = new int[actual.rows];
        for (int i = 0; i < actual.rows; i++) {
            float value = actual.data[i]; // cols=1, so data[i*1+0] = data[i]
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("target value at row " + i + " must be finite");
            }

            int asInt = (int) value;
            if (Math.abs(value - asInt) > 1e-6f) {
                throw new IllegalArgumentException(
                        "target value at row " + i + " must be an integer class id, got " + value);
            }
            y[i] = asInt;
        }
        return y;
    }

    /**
     * Builds a [n x 1] Tensor from int[] targets.
     */
    public static Tensor fromIntTargets(int[] targets) {
        Tensor t = new Tensor(targets.length, 1);
        for (int i = 0; i < targets.length; i++) {
            t.data[i] = targets[i]; // cols=1, so data[i*1+0] = data[i]
        }
        return t;
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
}