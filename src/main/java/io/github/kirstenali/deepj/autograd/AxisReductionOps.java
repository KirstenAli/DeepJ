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
