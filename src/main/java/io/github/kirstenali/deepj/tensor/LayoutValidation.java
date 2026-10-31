package io.github.kirstenali.deepj.tensor;

import java.util.Comparator;
import java.util.stream.IntStream;

final class LayoutValidation {
    private LayoutValidation() {}

    static void requireBounds(TensorShape shape, long[] strides, int offset, int storageElements) {
        if (strides.length != shape.rank()) throw new IllegalArgumentException("Stride rank must match tensor rank");
        if (offset < 0 || storageElements < 0 || offset > storageElements) {
            throw new IllegalArgumentException("Layout offset is outside its storage");
        }
        requireNonNegativeStrides(strides);
        if (shape.elements() == 0) return;
        long last = lastAddress(shape, strides, offset);
        if (last >= storageElements) throw new IllegalArgumentException("Tensor layout exceeds its storage");
    }

    private static void requireNonNegativeStrides(long[] strides) {
        for (long stride : strides) {
            if (stride < 0) throw new IllegalArgumentException("Negative strides are not supported");
        }
    }

    private static long lastAddress(TensorShape shape, long[] strides, int offset) {
        long address = offset;
        for (int axis = 0; axis < shape.rank(); axis++) {
            address = Math.addExact(address, Math.multiplyExact(strides[axis], shape.size(axis) - 1L));
        }
        return address;
    }

    static int index(int index, int size) {
        int normalized = index < 0 ? index + size : index;
        if (normalized < 0 || normalized >= size) throw new IndexOutOfBoundsException(index);
        return normalized;
    }

    /** Sufficient injectivity proof for dense, permuted, sliced and expanded layouts produced by our view API. */
    static boolean nonOverlapping(TensorShape shape, long[] strides) {
        int[] axes = IntStream.range(0, shape.rank()).filter(axis -> shape.size(axis) > 1).boxed()
                .sorted(Comparator.comparingLong(axis -> strides[axis])).mapToInt(Integer::intValue).toArray();
        long span = 1;
        for (int axis : axes) {
            if (strides[axis] < span) return false;
            span = Math.addExact(span, Math.multiplyExact(shape.size(axis) - 1L, strides[axis]));
        }
        return true;
    }
}
