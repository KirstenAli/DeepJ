package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.Objects;

/** Immutable logical dimensions. Rank zero has one value; a zero-length axis makes an empty tensor. */
final class TensorShape {
    private final int[] dimensions;
    private final int elements;

    TensorShape(int... dimensions) {
        this.dimensions = Objects.requireNonNull(dimensions, "dimensions").clone();
        elements = count(this.dimensions);
    }

    int rank() { return dimensions.length; }
    int elements() { return elements; }
    boolean matches(int[] expected) { return Arrays.equals(dimensions, expected); }
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

    private static int count(int[] dimensions) {
        long total = 1; // Above intMax is an overflow witness; a later zero must still reset it.
        for (int dimension : dimensions) {
            if (dimension < 0) throw new IllegalArgumentException("Tensor dimensions cannot be negative");
            if (dimension == 0) total = 0;
            else if (total <= Integer.MAX_VALUE) total *= dimension; // At most intMax squared, below longMax.
        }
        try { return Math.toIntExact(total); }
        catch (ArithmeticException error) {
            throw new IllegalArgumentException("Tensor element count exceeds the dense-storage limit", error);
        }
    }

    @Override public boolean equals(Object other) {
        return other instanceof TensorShape shape && Arrays.equals(dimensions, shape.dimensions);
    }

    @Override public int hashCode() { return Arrays.hashCode(dimensions); }

    @Override public String toString() { return Arrays.toString(dimensions); }
}
