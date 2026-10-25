package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TensorShapeTest {
    @Test
    void scalarEmptyAndDenseShapesHaveCheckedElementCounts() {
        assertEquals(1, new TensorShape().elements());
        assertEquals(0, new TensorShape(2, 0, 3).elements());
        assertEquals(0, new TensorShape(Integer.MAX_VALUE, Integer.MAX_VALUE, 0).elements());
        assertEquals(24, new TensorShape(2, 3, 4).elements());
        assertThrows(IllegalArgumentException.class, () -> new TensorShape(2, -1));
        assertThrows(IllegalArgumentException.class, () -> new TensorShape(0, -1));
        assertThrows(IllegalArgumentException.class, () -> new TensorShape(65536, 65536));
        assertThrows(NullPointerException.class, () -> new TensorShape((int[]) null));
    }

    @Test
    void dimensionsAreImmutableAndAxesNormalizeFromTheEnd() {
        int[] dimensions = {2, 3, 4};
        var shape = new TensorShape(dimensions);
        dimensions[0] = 99;
        shape.dimensions()[1] = 99;
        assertArrayEquals(new int[]{2, 3, 4}, shape.dimensions());
        assertEquals(3, shape.rank());
        assertEquals(4, shape.size(-1));
        assertEquals(0, shape.axis(-3));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.size(-4));
        assertThrows(IndexOutOfBoundsException.class, () -> shape.axis(3));
        assertThrows(IndexOutOfBoundsException.class, () -> new TensorShape().axis(0));
    }

    @Test
    void denseStridesAndEqualityAreValueBased() {
        var shape = new TensorShape(2, 3, 4);
        assertArrayEquals(new long[]{12, 4, 1}, shape.denseStrides());
        assertArrayEquals(new long[]{3, 3, 1}, new TensorShape(2, 0, 3).denseStrides());
        assertArrayEquals(new long[0], new TensorShape().denseStrides());
        assertEquals(shape, new TensorShape(2, 3, 4));
        assertEquals(shape.hashCode(), new TensorShape(2, 3, 4).hashCode());
        assertNotEquals(shape, new TensorShape(24));
        assertNotEquals(shape, "[2, 3, 4]");
        assertEquals("[2, 3, 4]", shape.toString());
        assertThrows(ArithmeticException.class,
                () -> new TensorShape(0, Integer.MAX_VALUE, Integer.MAX_VALUE, 3).denseStrides());
    }
}
