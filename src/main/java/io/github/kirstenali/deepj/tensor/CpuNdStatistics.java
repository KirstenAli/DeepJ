package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

/** Two-pass FLOAT32 statistics without a full-sized FLOAT32 shadow of compact input. */
public final class CpuNdStatistics {
    private CpuNdStatistics() {}

    public static Tensor variance(Tensor input, int[] axes, boolean keep, int correction) {
        if (correction < 0) throw new IllegalArgumentException("Variance correction cannot be negative");
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        float[] means = CpuNdReduction.sums(input, shape);
        for (int index = 0; index < means.length; index++) {
            means[index] /= shape.count();
        }
        float[] squares = centeredSquares(input, shape, means);
        Tensor output = new Tensor(input.dtype(), shape.output().dimensions());
        long denominator = shape.count() - correction;
        for (int index = 0; index < squares.length; index++) {
            output.memory().set(index, denominator > 0 ? squares[index] / denominator : Float.NaN);
        }
        return output;
    }

    private static float[] centeredSquares(Tensor input, ReductionShape shape, float[] means) {
        float[] result = new float[means.length];
        for (int index = 0; index < input.numel(); index++) {
            int target = shape.outputIndex(index);
            float value = input.memory().get(input.layout().addressAt(index)) - means[target];
            result[target] += value * value;
        }
        return result;
    }

    public static Tensor maximum(Tensor input, int[] axes, boolean keep) {
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        if (shape.count() == 0) throw new IllegalArgumentException("Maximum cannot reduce an empty axis");
        input.materialize();
        float[] maxima = new float[shape.output().elements()];
        Arrays.fill(maxima, Float.NEGATIVE_INFINITY);
        for (int index = 0; index < input.numel(); index++) {
            int target = shape.outputIndex(index);
            maxima[target] = Math.max(maxima[target], value(input, index));
        }
        Tensor output = new Tensor(input.dtype(), shape.output().dimensions());
        for (int index = 0; index < maxima.length; index++) {
            output.memory().set(index, maxima[index]);
        }
        return output;
    }

    public static Tensor maximumGradient(Tensor input, Tensor maximum, Tensor gradient, int[] axes, boolean keep) {
        var shape = new ReductionShape(input.layout().shape(), axes, keep);
        if (!maximum.layout().shape().equals(shape.output())) {
            throw new IllegalArgumentException("Invalid maximum shape");
        }
        Tensor.requireSameShape(maximum, gradient, "Maximum gradient");
        input.materialize();
        maximum.materialize();
        gradient.materialize();
        int[] counts = ties(input, maximum, shape);
        Tensor output = new Tensor(DType.FLOAT32, input.shape());
        for (int index = 0; index < output.numel(); index++) {
            int target = shape.outputIndex(index);
            float winner = value(maximum, target);
            output.memory().set(index,
                    derivative(value(input, index), winner, value(gradient, target), counts[target]));
        }
        return output;
    }

    private static int[] ties(Tensor input, Tensor maximum, ReductionShape shape) {
        int[] counts = new int[maximum.numel()];
        for (int index = 0; index < input.numel(); index++) {
            int target = shape.outputIndex(index);
            if (value(input, index) == value(maximum, target)) counts[target]++;
        }
        return counts;
    }

    private static float derivative(float input, float maximum, float gradient, int ties) {
        if (Float.isNaN(maximum)) return Float.NaN;
        return input == maximum ? gradient / ties : 0;
    }

    private static float value(Tensor input, int index) { return input.memory().get(input.layout().addressAt(index)); }
}
