package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.Tensor;

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

        // log-softmax + NLL
        double lossSum = 0.0;
        for (int i = 0; i < logits.rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int v = 0; v < logits.cols; v++) {
                double z = logits.data[i][v];
                if (z > max) max = z;
            }
            double sumExp = 0.0;
            for (int v = 0; v < logits.cols; v++) {
                sumExp += Math.exp(logits.data[i][v] - max);
            }
            double logDen = Math.log(sumExp) + max;

            int t = targets[i];
            lossSum += (logDen - logits.data[i][t]);
        }
        return lossSum / logits.rows;
    }

    /**
     * Convenience helper: gradient w.r.t. logits, averaged over rows.
     */
    public static Tensor gradient(Tensor logits, int[] targets) {
        checkTargets(logits, targets);

        Tensor grad = new Tensor(logits.rows, logits.cols);

        for (int i = 0; i < logits.rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int v = 0; v < logits.cols; v++) {
                double z = logits.data[i][v];
                if (z > max) max = z;
            }
            double sumExp = 0.0;
            for (int v = 0; v < logits.cols; v++) {
                sumExp += Math.exp(logits.data[i][v] - max);
            }

            for (int v = 0; v < logits.cols; v++) {
                double p = Math.exp(logits.data[i][v] - max) / sumExp;
                grad.data[i][v] = p;
            }
            // subtract 1 for the true class
            grad.data[i][targets[i]] -= 1.0;
        }

        // average
        return grad.divideScalar(logits.rows);
    }

    /**
     * Converts a [n x 1] Tensor of class indices into an int[].
     */
    public static int[] toIntTargets(Tensor actual) {
        if (actual.cols != 1) {
            throw new IllegalArgumentException("CrossEntropyLoss expects targets shape [n x 1], got [" + actual.rows + " x " + actual.cols + "]");
        }
        int[] y = new int[actual.rows];
        for (int i = 0; i < actual.rows; i++) {
            y[i] = (int) Math.round(actual.data[i][0]);
        }
        return y;
    }

    /**
     * Builds a [n x 1] Tensor from int[] targets.
     */
    public static Tensor fromIntTargets(int[] targets) {
        Tensor t = new Tensor(targets.length, 1);
        for (int i = 0; i < targets.length; i++) t.data[i][0] = targets[i];
        return t;
    }

    private static void checkTargets(Tensor logits, int[] targets) {
        if (logits == null) throw new IllegalArgumentException("logits is null");
        if (targets == null) throw new IllegalArgumentException("targets is null");
        if (targets.length != logits.rows) {
            throw new IllegalArgumentException("targets length " + targets.length + " must match logits rows " + logits.rows);
        }
        for (int t : targets) {
            if (t < 0 || t >= logits.cols) {
                throw new IllegalArgumentException("target id out of range: " + t + " (vocab=" + logits.cols + ")");
            }
        }
    }
}
