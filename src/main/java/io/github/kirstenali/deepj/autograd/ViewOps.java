package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.stream.IntStream;

public final class ViewOps {
    private ViewOps() {}

    public static Tensor unsqueeze(Tensor input, int axis) {
        int[] original = input.shape();
        return UnaryOps.record(input, Tensor.backend().unsqueeze(input, axis),
                gradient -> Tensor.backend().reshape(gradient, original));
    }

    public static Tensor view(Tensor input, int... dimensions) {
        int[] original = input.shape();
        return UnaryOps.record(input, Tensor.backend().view(input, dimensions),
                gradient -> Tensor.backend().reshape(gradient, original));
    }

    public static Tensor reshape(Tensor input, int... dimensions) {
        int[] original = input.shape();
        return UnaryOps.record(input, Tensor.backend().reshape(input, dimensions),
                gradient -> Tensor.backend().reshape(gradient, original));
    }

    public static Tensor permute(Tensor input, int... axes) {
        Tensor output = Tensor.backend().permute(input, axes);
        int[] inverse = new int[axes.length];
        for (int axis = 0; axis < axes.length; axis++) {
            inverse[axis(input, axes[axis])] = axis;
        }
        return UnaryOps.record(input, output, gradient -> Tensor.backend().permute(gradient, inverse));
    }

    public static Tensor transpose(Tensor input, int first, int second) {
        int[] axes = IntStream.range(0, input.rank()).toArray();
        int a = axis(input, first), b = axis(input, second);
        axes[a] = b;
        axes[b] = a;
        return permute(input, axes);
    }

    public static Tensor expand(Tensor input, int... dimensions) {
        int[] original = input.shape();
        return UnaryOps.record(input, Tensor.backend().expand(input, dimensions),
                gradient -> Tensor.backend().sumToShape(gradient, original));
    }

    public static Tensor select(Tensor input, int dimension, int index) {
        int axis = axis(input, dimension);
        Tensor output = Tensor.backend().select(input, axis, index);
        int start = index < 0 ? index + input.size(axis) : index;
        int[] original = input.shape(), region = original.clone();
        region[axis] = 1;
        return UnaryOps.record(input, output, gradient -> Tensor.backend().sliceGradient(
                Tensor.backend().reshape(gradient, region), original, axis, start, 1));
    }

    public static Tensor slice(Tensor input, int dimension, int start, int stop, int step) {
        int axis = axis(input, dimension);
        int first = bound(start, input.size(axis)), last = Math.max(first, bound(stop, input.size(axis)));
        int[] original = input.shape();
        Tensor output = Tensor.backend().slice(input, axis, first, last, step);
        return UnaryOps.record(input, output,
                gradient -> Tensor.backend().sliceGradient(gradient, original, axis, first, step));
    }

    public static Tensor contiguous(Tensor input) {
        if (input.isContiguous()) return input;
        return copy(input);
    }

    public static Tensor copy(Tensor input) {
        return UnaryOps.record(input, Tensor.backend().copy(input), gradient -> gradient);
    }

    public static Tensor cast(Tensor input, DType dtype) {
        if (input.dtype() == dtype) return input;
        return UnaryOps.record(input, Tensor.backend().cast(input, dtype), gradient -> gradient);
    }

    private static int axis(Tensor input, int dimension) {
        input.size(dimension);
        return dimension < 0 ? dimension + input.rank() : dimension;
    }

    private static int bound(int index, int size) {
        return Math.max(0, Math.min(size, index < 0 ? index + size : index));
    }
}
