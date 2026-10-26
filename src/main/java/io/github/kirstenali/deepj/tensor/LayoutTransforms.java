package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

final class LayoutTransforms {
    private LayoutTransforms() {}

    static TensorLayout unsqueeze(TensorLayout input, int axis) {
        int rank = input.shape().rank();
        axis = axis < 0 ? axis + rank + 1 : axis;
        if (axis < 0 || axis > rank) throw new IndexOutOfBoundsException("Insertion axis: " + axis);
        int[] dimensions = Arrays.copyOf(input.shape().dimensions(), rank + 1);
        long[] strides = Arrays.copyOf(input.strides(), rank + 1);
        System.arraycopy(dimensions, axis, dimensions, axis + 1, rank - axis);
        System.arraycopy(strides, axis, strides, axis + 1, rank - axis);
        dimensions[axis] = 1;
        strides[axis] = 0;
        return input.with(new TensorShape(dimensions), strides, input.offset());
    }

    static TensorLayout permute(TensorLayout input, int... axes) {
        if (axes.length != input.shape().rank()) throw new IllegalArgumentException("Permutation rank must match");
        int[] dimensions = new int[axes.length];
        long[] strides = new long[axes.length];
        boolean[] seen = new boolean[axes.length];
        for (int axis = 0; axis < axes.length; axis++) {
            int source = input.shape().axis(axes[axis]);
            if (seen[source]) throw new IllegalArgumentException("Permutation axes cannot repeat");
            seen[source] = true;
            dimensions[axis] = input.shape().size(source);
            strides[axis] = input.stride(source);
        }
        return input.with(new TensorShape(dimensions), strides, input.offset());
    }

    static TensorLayout view(TensorLayout input, int... dimensions) {
        if (!input.contiguous()) throw new IllegalArgumentException("View requires contiguous storage; use reshape");
        TensorShape shape = ShapeTransforms.reshape(input.shape(), dimensions);
        return input.with(shape, shape.denseStrides(), input.offset());
    }

    static TensorLayout expand(TensorLayout input, int... dimensions) {
        TensorShape shape = new TensorShape(dimensions);
        if (shape.rank() < input.shape().rank() || !ShapeTransforms.broadcast(input.shape(), shape).equals(shape)) {
            throw new IllegalArgumentException("Expansion shape must broadcast from the original shape");
        }
        long[] strides = new long[shape.rank()];
        for (int axis = 0; axis < shape.rank(); axis++) {
            int source = axis - shape.rank() + input.shape().rank();
            strides[axis] = source < 0 || input.shape().size(source) == 1 ? 0 : input.stride(source);
        }
        return input.with(shape, strides, input.offset());
    }

    static TensorLayout select(TensorLayout input, int axis, int index) {
        axis = input.shape().axis(axis);
        index = LayoutValidation.index(index, input.shape().size(axis));
        int[] dimensions = without(input.shape().dimensions(), axis);
        long[] strides = without(input.strides(), axis);
        int offset = Math.toIntExact(input.offset() + input.stride(axis) * index);
        return input.with(new TensorShape(dimensions), strides, offset);
    }

    static TensorLayout slice(TensorLayout input, int axis, int start, int stop, int step) {
        axis = input.shape().axis(axis);
        int size = input.shape().size(axis);
        if (step < 1) throw new IllegalArgumentException("Slice step must be positive");
        if (start < 0 || stop < start || stop > size) throw new IndexOutOfBoundsException("Invalid slice bounds");
        int[] dimensions = input.shape().dimensions();
        dimensions[axis] = (int) (((long) stop - start + step - 1) / step);
        long[] strides = input.strides();
        strides[axis] = Math.multiplyExact(strides[axis], step);
        TensorShape shape = new TensorShape(dimensions);
        int offset = shape.elements() == 0 ? input.offset()
                : Math.toIntExact(input.offset() + input.stride(axis) * start);
        return input.with(shape, strides, offset);
    }

    private static int[] without(int[] values, int axis) {
        int[] result = Arrays.copyOf(values, values.length - 1);
        System.arraycopy(values, axis + 1, result, axis, result.length - axis);
        return result;
    }

    private static long[] without(long[] values, int axis) {
        long[] result = Arrays.copyOf(values, values.length - 1);
        System.arraycopy(values, axis + 1, result, axis, result.length - axis);
        return result;
    }
}
