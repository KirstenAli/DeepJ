package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

final class DenseLayoutConstructionTest {
    private static final int[] EDGES = {-1, 0, 1, 2, 3, 65536, Integer.MAX_VALUE};

    @Test
    void canonicalConstructionMatchesIndependentIntegerBoundsAndFailureOracle() {
        Random random = new Random(55);
        for (int rank = 0; rank <= 7; rank++) {
            for (int trial = 0; trial < 1000; trial++) {
                verify(randomShape(random, rank));
            }
        }
    }

    @Test
    void generalViewsStillRejectInvalidRankOffsetStrideExtentAndOverflow() {
        reject(new int[]{2, 3}, new long[]{3, 1}, 1, 6, IllegalArgumentException.class);
        reject(new int[]{2, 3}, new long[]{3, 1}, 0, 5, IllegalArgumentException.class);
        reject(new int[]{2, 3}, new long[]{3, -1}, 0, 6, IllegalArgumentException.class);
        reject(new int[]{2, 3}, new long[]{3}, 0, 6, IllegalArgumentException.class);
        reject(new int[]{2, 3}, new long[]{Long.MAX_VALUE, 1}, 0, 6, ArithmeticException.class);
        reject(new int[]{0, 3}, new long[]{3, 1}, 1, 0, IllegalArgumentException.class);
        assertThrows(NullPointerException.class, () -> TensorLayout.dense(null));
    }

    private static int[] randomShape(Random random, int rank) {
        int[] dimensions = new int[rank];
        for (int axis = 0; axis < rank; axis++) { dimensions[axis] = EDGES[random.nextInt(EDGES.length)]; }
        return dimensions;
    }

    private static void verify(int[] dimensions) {
        Expected expected = expected(dimensions);
        if (expected.failure() != null) {
            assertThrows(expected.failure(), () -> TensorLayout.dense(new TensorShape(dimensions)));
            return;
        }
        TensorLayout layout = TensorLayout.dense(new TensorShape(dimensions));
        assertEquals(0, layout.offset());
        assertEquals(expected.elements(), layout.storageElements());
        assertArrayEquals(expected.strides(), layout.strides());
        assertArrayEquals(dimensions, layout.shape().dimensions());
        assertTrue(layout.contiguous());
        if (expected.elements() > 0) {
            assertEquals(BigInteger.valueOf(expected.elements() - 1L), lastAddress(dimensions, layout.strides()));
        }
    }

    private static Expected expected(int[] dimensions) {
        if (Arrays.stream(dimensions).anyMatch(value -> value < 0)) {
            return new Expected(0, null, IllegalArgumentException.class);
        }
        BigInteger count = product(dimensions);
        if (count.compareTo(BigInteger.valueOf(Integer.MAX_VALUE)) > 0) {
            return new Expected(0, null, IllegalArgumentException.class);
        }
        try { return new Expected(count.intValueExact(), strides(dimensions), null); }
        catch (ArithmeticException error) { return new Expected(0, null, ArithmeticException.class); }
    }

    private static BigInteger product(int[] dimensions) {
        BigInteger product = BigInteger.ONE;
        for (int dimension : dimensions) { product = product.multiply(BigInteger.valueOf(dimension)); }
        return product;
    }

    private static long[] strides(int[] dimensions) {
        long[] strides = new long[dimensions.length];
        BigInteger next = BigInteger.ONE;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            strides[axis] = next.longValueExact();
            next = next.multiply(BigInteger.valueOf(dimensions[axis] == 0 ? 1 : dimensions[axis]));
            next.longValueExact();
        }
        return strides;
    }

    private static BigInteger lastAddress(int[] dimensions, long[] strides) {
        BigInteger last = BigInteger.ZERO;
        for (int axis = 0; axis < dimensions.length; axis++) {
            last = last.add(BigInteger.valueOf(strides[axis]).multiply(BigInteger.valueOf(dimensions[axis] - 1L)));
        }
        return last;
    }

    private static void reject(int[] dimensions, long[] strides, int offset, int storage,
                               Class<? extends RuntimeException> failure) {
        assertThrows(failure, () -> new TensorLayout(new TensorShape(dimensions), strides, offset, storage));
    }

    private record Expected(int elements, long[] strides, Class<? extends RuntimeException> failure) { }
}
