package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TypedArrayTest {
    @Test
    void compactAllocationsHaveOnlyTwoBytesPerElement() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var array = new TypedArray(dtype, 100);
            assertEquals(dtype, array.dtype());
            assertEquals(100, array.length());
            assertEquals(200, array.byteLength());
            assertEquals(100, array.shorts().length);
            assertThrows(IllegalStateException.class, array::floats);
            array.set(23, 1.5f);
            assertEquals(1.5f, array.get(23));
        }
    }

    @Test
    void float32FastPathKeepsItsUnroundedValues() {
        var array = new TypedArray(DType.FLOAT32, 3);
        array.set(1, 1.0000001f);
        assertEquals(1.0000001f, array.get(1));
        assertEquals(1.0000001f, array.floats()[1]);
        assertEquals(12, array.byteLength());
        assertThrows(IllegalStateException.class, array::shorts);
    }

    @Test
    void rawCompactBitsCanBePreservedForTransfersAndCheckpointing() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var array = new TypedArray(dtype, 65536);
            for (int bits = 0; bits <= 0xffff; bits++) {
                array.shorts()[bits] = (short) bits;
            }
            assertEquals(131072, array.byteLength());
            for (int bits = 0; bits <= 0xffff; bits++) {
                assertEquals((short) bits, array.shorts()[bits]);
            }
        }
    }

    @Test
    void emptyAndInvalidAllocationsAreExplicit() {
        assertEquals(0, new TypedArray(DType.FLOAT16, 0).shorts().length);
        assertThrows(IllegalArgumentException.class, () -> new TypedArray(DType.FLOAT32, -1));
        assertThrows(NullPointerException.class, () -> new TypedArray(null, 1));
        var array = new TypedArray(DType.BFLOAT16, 1);
        assertThrows(IndexOutOfBoundsException.class, () -> array.get(1));
        assertThrows(IndexOutOfBoundsException.class, () -> array.set(-1, 1));
    }
}
