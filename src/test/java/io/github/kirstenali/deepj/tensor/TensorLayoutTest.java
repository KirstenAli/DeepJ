package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TensorLayoutTest {
    @Test
    void denseLayoutsDoNotExposeOrShareTheirFreshStrideArrays() {
        var shape = new TensorShape(2, 3, 4);
        var first = TensorLayout.dense(shape);
        var second = TensorLayout.dense(shape);
        first.strides()[1] = 999;
        shape.denseStrides()[0] = 999;
        assertArrayEquals(new long[]{12, 4, 1}, first.strides());
        assertArrayEquals(first.strides(), second.strides());
        assertEquals(23, first.addressAt(23));
        assertThrows(NullPointerException.class, () -> TensorLayout.dense(null));
    }

    @Test
    void viewConstructionStillDefensivelyCopiesSuppliedStrides() {
        var dense = TensorLayout.dense(new TensorShape(2, 3));
        long[] strides = {1, 3};
        var view = dense.with(new TensorShape(3, 2), strides, 0);
        strides[0] = 999;
        assertArrayEquals(new long[]{1, 3}, view.strides());
        assertEquals(5, view.address(2, 1));
        assertFalse(view.contiguous());
    }

    @Test
    void scalarAndEmptyLayoutsHaveExplicitAddressingRules() {
        var scalar = TensorLayout.dense(new TensorShape());
        assertEquals(0, scalar.address());
        assertEquals(0, scalar.addressAt(0));
        assertTrue(scalar.contiguous());
        assertTrue(scalar.nonOverlapping());
        var empty = new TensorLayout(new TensorShape(0, 3), new long[]{3, 1}, 8, 8);
        assertTrue(empty.contiguous());
        assertDoesNotThrow(empty::requireWritable);
        assertThrows(IndexOutOfBoundsException.class, () -> empty.addressAt(0));
        assertThrows(IndexOutOfBoundsException.class, () -> empty.address(0, 0));
    }

    @Test
    void coordinatesAndLogicalIndicesAddressTheSameElements() {
        var layout = TensorLayout.dense(new TensorShape(2, 3, 4));
        assertEquals(23, layout.address(-1, -1, -1));
        assertEquals(17, layout.address(1, 1, 1));
        assertEquals(17, layout.addressAt(17));
        assertEquals(0, layout.offset());
        assertEquals(24, layout.storageElements());
        assertThrows(IllegalArgumentException.class, () -> layout.address(1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> layout.address(2, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> layout.address(-3, 0, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> layout.addressAt(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> layout.addressAt(24));
    }

    @Test
    void stridesAreImmutableAndSingletonAxesDoNotPreventContiguity() {
        long[] strides = {4, 99, 1};
        var layout = new TensorLayout(new TensorShape(2, 1, 4), strides, 0, 8);
        strides[2] = 50;
        layout.strides()[0] = 50;
        assertArrayEquals(new long[]{4, 99, 1}, layout.strides());
        assertEquals(1, layout.stride(-1));
        assertTrue(layout.contiguous());
        assertDoesNotThrow(layout::requireWritable);
    }

    @Test
    void overlappingViewsAreReadableButNotWritable() {
        var overlap = new TensorLayout(new TensorShape(2, 2), new long[]{1, 1}, 0, 3);
        assertEquals(overlap.address(0, 1), overlap.address(1, 0));
        assertFalse(overlap.contiguous());
        assertFalse(overlap.nonOverlapping());
        assertThrows(IllegalStateException.class, overlap::requireWritable);
        var stepped = new TensorLayout(new TensorShape(4), new long[]{2}, 0, 8);
        assertFalse(stepped.contiguous());
        assertTrue(stepped.nonOverlapping());
        assertDoesNotThrow(stepped::requireWritable);
    }

    @Test
    void constructionRejectsInvalidStorageSpansAndOverflow() {
        var shape = new TensorShape(2, 2);
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2}, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2, 1}, -1, 4));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2, 1}, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2, 1}, 5, 4));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2, -1}, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> new TensorLayout(shape, new long[]{2, 1}, 1, 4));
        assertThrows(ArithmeticException.class,
                () -> new TensorLayout(shape, new long[]{Long.MAX_VALUE, Long.MAX_VALUE}, 1, 4));
        assertThrows(NullPointerException.class, () -> new TensorLayout(null, new long[0], 0, 0));
        assertThrows(NullPointerException.class, () -> new TensorLayout(shape, null, 0, 4));
    }
}
