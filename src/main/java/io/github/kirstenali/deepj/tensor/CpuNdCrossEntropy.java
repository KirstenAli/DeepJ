package io.github.kirstenali.deepj.tensor;

/** Stable final-axis loss and FLOAT32 gradient, including strided/compact logits and response masks. */
public final class CpuNdCrossEntropy {
    private CpuNdCrossEntropy() {}

    public static CrossEntropyResult apply(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyChecks.requireTargets(logits, targets);
        Tensor gradient = new Tensor(DType.FLOAT32, logits.shape());
        float value = compute(logits, targets, mask, gradient);
        int[] scalarShape = logits.rank() == 2 ? new int[]{1, 1} : new int[0];
        Tensor loss = Tensor.from(new float[]{value}, scalarShape);
        return new CrossEntropyResult(loss, gradient);
    }

    public static float loss(Tensor logits, int[] targets, boolean[] mask) {
        return compute(logits, targets, mask, null);
    }

    private static float compute(Tensor logits, int[] targets, boolean[] mask, Tensor gradient) {
        CrossEntropyChecks.requireTargets(logits, targets);
        int positions = CrossEntropyChecks.positions(logits), classes = CrossEntropyChecks.classes(logits);
        int included = mask == null ? positions : CrossEntropyChecks.included(mask, positions);
        logits.materialize();
        float sum = 0;
        for (int row = 0; row < positions; row++) {
            if (mask == null || mask[row]) sum += row(logits, gradient, row * classes, targets[row], classes, included);
        }
        return sum / included;
    }

    private static float row(Tensor logits, Tensor gradient, int base, int target, int classes, int included) {
        float maximum = Float.NEGATIVE_INFINITY;
        for (int column = 0; column < classes; column++) {
            maximum = Math.max(maximum, value(logits, base + column));
        }
        float sum = 0;
        for (int column = 0; column < classes; column++) {
            sum += exponential(value(logits, base + column) - maximum);
        }
        if (gradient != null) writeGradient(logits, gradient, base, target, classes, included, maximum, sum);
        return (float) Math.log(sum) + maximum - value(logits, base + target);
    }

    private static void writeGradient(Tensor logits, Tensor gradient, int base, int target,
                                        int classes, int included, float maximum, float sum) {
        for (int column = 0; column < classes; column++) {
            float probability = exponential(value(logits, base + column) - maximum) / sum;
            float derivative = probability - (column == target ? 1.0f : 0.0f);
            gradient.memory().set(base + column, derivative / included);
        }
    }

    private static float value(Tensor tensor, int index) {
        return tensor.memory().get(tensor.layout().addressAt(index));
    }
    private static float exponential(float value) { return (float) Math.exp(value); }
}
