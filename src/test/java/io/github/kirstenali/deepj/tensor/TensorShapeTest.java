package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TensorShapeTest {
    @Test
    void overflowWitnessPreservesCauseAndLaterZeroOrNegativePrecedence() {
        int largest = Integer.MAX_VALUE;
        var overflow = assertThrows(IllegalArgumentException.class, () -> new TensorShape(largest, 2, 3));
        assertInstanceOf(ArithmeticException.class, overflow.getCause());
        assertEquals("integer overflow", overflow.getCause().getMessage());
        assertEquals(0, new TensorShape(largest, 2, 0).elements());
        var invalid = assertThrows(IllegalArgumentException.class, () -> new TensorShape(largest, 2, 0, -1));
        assertEquals("Tensor dimensions cannot be negative", invalid.getMessage());
        assertNull(invalid.getCause());
    }

    @Test
    void zeroNeverHidesAnInvalidDimensionAtAnyOtherPosition() {
        for (int negative = 0; negative < 6; negative++) {
            for (int zero = 0; zero < 6; zero++) {
                if (negative == zero) continue;
                int[] dimensions = {3, 3, 3, 3, 3, 3};
                dimensions[negative] = -1;
                dimensions[zero] = 0;
                var error = assertThrows(IllegalArgumentException.class, () -> new TensorShape(dimensions));
                assertEquals("Tensor dimensions cannot be negative", error.getMessage());
            }
        }
    }

    @Test
    void emptyCountRetainsFullValidationAndIndependentStrideOverflow() {
        int largest = Integer.MAX_VALUE;
        assertEquals(0, new TensorShape(largest, largest, 0).elements());
        assertArrayEquals(new long[]{(long) largest * largest, largest, 1},
                new TensorShape(0, largest, largest).denseStrides());
        var error = assertThrows(IllegalArgumentException.class,
                () -> new TensorShape(largest, largest, 0, -1));
        assertEquals("Tensor dimensions cannot be negative", error.getMessage());
        assertThrows(ArithmeticException.class,
                () -> new TensorShape(0, largest, largest, largest).denseStrides());
    }

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

    @Test
    void equalBroadcastShapesReuseOnlyImmutableMetadata() {
        int[] dimensions = {2, 3, 4};
        var left = new TensorShape(dimensions);
        var right = new TensorShape(2, 3, 4);
        var result = ShapeTransforms.broadcast(left, right);
        assertSame(left, result);
        dimensions[0] = 99;
        result.dimensions()[1] = 99;
        assertArrayEquals(new int[]{2, 3, 4}, right.dimensions());
        assertArrayEquals(new int[]{2, 3, 4}, result.dimensions());
        assertSame(left, ShapeTransforms.broadcast(left, left));
        assertSame(result, ShapeTransforms.broadcast(result, new TensorShape(2, 3, 4)));
    }

    @Test
    void scalarAndEmptyBroadcastFastPathsPreserveValidationAndRank() {
        var scalar = new TensorShape();
        var empty = new TensorShape(2, 0, 3);
        assertSame(scalar, ShapeTransforms.broadcast(scalar, new TensorShape()));
        assertSame(empty, ShapeTransforms.broadcast(empty, new TensorShape(2, 0, 3)));
        assertEquals(empty, ShapeTransforms.broadcast(empty, new TensorShape(1, 1, 3)));
        assertEquals(new TensorShape(2, 3, 4), ShapeTransforms.broadcast(scalar, new TensorShape(2, 3, 4)));
        assertThrows(IllegalArgumentException.class,
                () -> ShapeTransforms.broadcast(empty, new TensorShape(2, 5, 3)));
        assertThrows(IllegalArgumentException.class, () -> new TensorShape(Integer.MAX_VALUE, 0, -1));
    }
}
