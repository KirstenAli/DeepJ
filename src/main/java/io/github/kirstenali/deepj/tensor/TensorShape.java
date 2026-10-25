package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.Objects;

/** Immutable logical dimensions. Rank zero has one value; a zero-length axis makes an empty tensor. */
final class TensorShape {
    private final int[] dimensions;
    private final int elements;

    TensorShape(int... dimensions) {
        this.dimensions = Objects.requireNonNull(dimensions, "dimensions").clone();
        validate(this.dimensions);
        elements = count(this.dimensions);
    }

    int rank() { return dimensions.length; }
    int elements() { return elements; }
    int[] dimensions() { return dimensions.clone(); }
    int size(int axis) { return dimensions[axis(axis)]; }

    int axis(int axis) {
        int normalized = axis < 0 ? axis + rank() : axis;
        if (normalized < 0 || normalized >= rank()) throw new IndexOutOfBoundsException("Axis: " + axis);
        return normalized;
    }

    long[] denseStrides() {
        long[] strides = new long[rank()];
        long stride = 1;
        for (int axis = rank() - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, Math.max(1, dimensions[axis]));
        }
        return strides;
    }

    private static void validate(int[] dimensions) {
        for (int dimension : dimensions) {
            if (dimension < 0) throw new IllegalArgumentException("Tensor dimensions cannot be negative");
        }
    }

    private static int count(int[] dimensions) {
        if (Arrays.stream(dimensions).anyMatch(dimension -> dimension == 0)) return 0;
        int total = 1;
        try {
            for (int dimension : dimensions) {
                total = Math.multiplyExact(total, dimension);
            }
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Tensor element count exceeds the dense-storage limit", error);
        }
        return total;
    }

    @Override public boolean equals(Object other) {
        return other instanceof TensorShape shape && Arrays.equals(dimensions, shape.dimensions);
    }

    @Override public int hashCode() { return Arrays.hashCode(dimensions); }

    @Override public String toString() { return Arrays.toString(dimensions); }
}
