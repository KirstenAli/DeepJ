package io.github.kirstenali.deepj.tensor;

import java.util.Objects;

/** Immutable logical layout over one physical allocation. Strides and offset count elements, not bytes. */
final class TensorLayout {
    private final TensorShape shape;
    private final long[] strides;
    private final int offset, storageElements;

    TensorLayout(TensorShape shape, long[] strides, int offset, int storageElements) {
        this.shape = Objects.requireNonNull(shape, "shape");
        this.strides = Objects.requireNonNull(strides, "strides").clone();
        this.offset = offset;
        this.storageElements = storageElements;
        LayoutValidation.requireBounds(shape, this.strides, offset, storageElements);
    }

    static TensorLayout dense(TensorShape shape) {
        return new TensorLayout(shape, shape.denseStrides(), 0, shape.elements());
    }

    TensorShape shape() { return shape; }
    int offset() { return offset; }
    int storageElements() { return storageElements; }
    long stride(int axis) { return strides[shape.axis(axis)]; }
    long[] strides() { return strides.clone(); }

    int address(int row, int column) {
        if (shape.rank() != 2) throw new IllegalArgumentException("Matrix indexing requires rank two");
        return Math.toIntExact(offset + strides[0] * LayoutValidation.index(row, shape.size(0))
                + strides[1] * LayoutValidation.index(column, shape.size(1)));
    }

    int address(int... indices) {
        if (indices.length != shape.rank()) throw new IllegalArgumentException("Index rank must match tensor rank");
        long address = offset;
        for (int axis = 0; axis < indices.length; axis++) {
            address += strides[axis] * LayoutValidation.index(indices[axis], shape.size(axis));
        }
        return Math.toIntExact(address);
    }

    int addressAt(int logicalIndex) {
        if (logicalIndex < 0 || logicalIndex >= shape.elements()) throw new IndexOutOfBoundsException(logicalIndex);
        long address = offset;
        for (int axis = shape.rank() - 1; axis >= 0; axis--) {
            address += strides[axis] * (logicalIndex % shape.size(axis));
            logicalIndex /= shape.size(axis);
        }
        return Math.toIntExact(address);
    }

    boolean contiguous() {
        if (shape.elements() == 0) return true;
        long expected = 1;
        for (int axis = shape.rank() - 1; axis >= 0; axis--) {
            if (shape.size(axis) > 1 && strides[axis] != expected) return false;
            expected *= shape.size(axis);
        }
        return true;
    }

    boolean nonOverlapping() {
        return contiguous() || LayoutValidation.nonOverlapping(shape, strides);
    }

    void requireWritable() {
        if (!nonOverlapping()) throw new IllegalStateException("Overlapping tensor views cannot be written");
    }

    TensorLayout with(TensorShape dimensions, long[] steps, int start) {
        return new TensorLayout(dimensions, steps, start, storageElements);
    }
}
