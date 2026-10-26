package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Independent double-precision, per-position reference; no Tensor reduction/softmax operators. */
final class NdCrossEntropyReference {
    private NdCrossEntropyReference() {}

    static double loss(Tensor logits, int[] targets, boolean[] mask) {
        float[] values = logits.toFloatArray();
        int width = logits.size(-1), count = count(mask);
        double result = 0;
        for (int row = 0; row < targets.length; row++) {
            if (mask[row]) result += rowLoss(values, row * width, width, targets[row]);
        }
        return result / count;
    }

    static float[] gradient(Tensor logits, int[] targets, boolean[] mask) {
        float[] values = logits.toFloatArray(), result = new float[values.length];
        int width = logits.size(-1), count = count(mask);
        for (int row = 0; row < targets.length; row++) {
            if (mask[row]) rowGradient(values, result, row * width, width, targets[row], count);
        }
        return result;
    }

    private static void rowGradient(float[] values, float[] result, int base, int width, int target, int count) {
        double maximum = maximum(values, base, width), sum = exponentials(values, base, width, maximum);
        for (int column = 0; column < width; column++) {
            double probability = Math.exp(values[base + column] - maximum) / sum;
            result[base + column] = (float) ((probability - (column == target ? 1 : 0)) / count);
        }
    }

    private static double rowLoss(float[] values, int base, int width, int target) {
        double maximum = maximum(values, base, width);
        return Math.log(exponentials(values, base, width, maximum)) + maximum - values[base + target];
    }

    private static double exponentials(float[] values, int base, int width, double maximum) {
        double result = 0;
        for (int column = 0; column < width; column++) {
            result += Math.exp(values[base + column] - maximum);
        }
        return result;
    }

    private static double maximum(float[] values, int base, int width) {
        double result = Double.NEGATIVE_INFINITY;
        for (int column = 0; column < width; column++) {
            result = Math.max(result, values[base + column]);
        }
        return result;
    }

    private static int count(boolean[] mask) {
        int result = 0;
        for (boolean included : mask) {
            if (included) result++;
        }
        return result;
    }
}
