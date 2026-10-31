package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class AxisReductionOps {
    private AxisReductionOps() {}

    public static Tensor sum(Tensor input, int[] axes, boolean keep) {
        int[] original = input.shape(), retained = retained(input, axes);
        Tensor output = Tensor.backend().sumAxes(input, axes, keep);
        return UnaryOps.record(input, output, gradient -> Tensor.backend().expand(
                Tensor.backend().reshape(gradient, retained), original));
    }

    public static Tensor softmax(Tensor input, int axis) {
        Tensor output = Tensor.backend().softmaxAxis(input, axis);
        return UnaryOps.record(input, output,
                gradient -> Tensor.backend().softmaxAxisBackward(gradient, output, axis), output);
    }

    public static Tensor mean(Tensor input, int[] axes, boolean keep) {
        Tensor output = Tensor.backend().meanAxes(input, axes, keep);
        int[] original = input.shape(), retained = retained(input, axes);
        float count = elements(input, axes);
        return UnaryOps.record(input, output, gradient -> Tensor.backend().divideScalar(
                expanded(gradient, original, retained), count));
    }

    public static Tensor variance(Tensor input, int[] axes, boolean keep, int correction) {
        int[] savedAxes = axes.clone();
        Tensor output = Tensor.backend().varianceAxes(input, savedAxes, keep, correction);
        float count = elements(input, savedAxes) - correction;
        return UnaryOps.record(input, output,
                gradient -> varianceGradient(input, gradient, savedAxes, count), input);
    }

    private static Tensor varianceGradient(Tensor input, Tensor gradient, int[] axes, float count) {
        Tensor value = GradientDType.accumulation(input);
        Tensor center = Tensor.backend().subtract(value, Tensor.backend().meanAxes(value, axes, true));
        Tensor upstream = expanded(gradient, input.shape(), retained(input, axes));
        return Tensor.backend().multiplyScalar(Tensor.backend().multiply(center, upstream),
                count > 0 ? 2 / count : Float.NaN);
    }

    public static Tensor maximum(Tensor input, int[] axes, boolean keep) {
        int[] savedAxes = axes.clone();
        Tensor output = Tensor.backend().maximumAxes(input, savedAxes, keep);
        return UnaryOps.record(input, output,
                gradient -> Tensor.backend().maximumAxesBackward(input, output, gradient, savedAxes, keep),
                input, output);
    }

    private static Tensor expanded(Tensor gradient, int[] original, int[] retained) {
        return Tensor.backend().expand(Tensor.backend().reshape(gradient, retained), original);
    }

    private static float elements(Tensor input, int[] axes) {
        if (axes.length == 0) return input.numel();
        long result = 1;
        for (int axis : axes) {
            result = Math.multiplyExact(result, input.size(axis));
        }
        return result;
    }

    private static int[] retained(Tensor input, int[] axes) {
        int[] shape = input.shape();
        if (axes.length == 0) java.util.Arrays.fill(shape, 1);
        for (int axis : axes) {
            input.size(axis);
            shape[axis < 0 ? axis + input.rank() : axis] = 1;
        }
        return shape;
    }
}
