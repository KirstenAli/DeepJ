package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.stream.IntStream;

/** The adjoint of trailing-axis broadcasting, preserving logical rather than physical shape. */
final class GradientShapes {
    private GradientShapes() {}

    static Tensor sumTo(TensorBackend backend, Tensor input, int... dimensions) {
        if (Arrays.equals(input.shape(), dimensions)) return input;
        TensorShape target = new TensorShape(dimensions);
        if (!ShapeTransforms.broadcast(target, input.layout().shape()).equals(input.layout().shape())) {
            throw new IllegalArgumentException("Gradient cannot reduce to the requested shape");
        }
        int leading = input.rank() - target.rank();
        if (leading < 0) throw new IllegalArgumentException("Gradient target rank cannot increase");
        int[] axes = IntStream.range(0, input.rank()).filter(axis -> axis < leading
                || target.size(axis - leading) == 1 && input.size(axis) != 1).toArray();
        return backend.reshape(backend.sumAxes(input, axes, true), dimensions);
    }
}
