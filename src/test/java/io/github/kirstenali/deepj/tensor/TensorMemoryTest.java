package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class TensorMemoryTest {
    @Test
    void compactAllocationsHaveOnlyTwoBytesPerElement() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var array = new TensorMemory(dtype, 100);
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
        var array = new TensorMemory(DType.FLOAT32, 3);
        array.set(1, 1.0000001f);
        assertEquals(1.0000001f, array.get(1));
        assertEquals(1.0000001f, array.floats()[1]);
        assertEquals(12, array.byteLength());
        assertThrows(IllegalStateException.class, array::shorts);
    }

    @Test
    void rawCompactBitsCanBePreservedForTransfersAndCheckpointing() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            var array = new TensorMemory(dtype, 65536);
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
        assertEquals(0, new TensorMemory(DType.FLOAT16, 0).shorts().length);
        assertThrows(IllegalArgumentException.class, () -> new TensorMemory(DType.FLOAT32, -1));
        assertThrows(NullPointerException.class, () -> new TensorMemory(null, 1));
        var array = new TensorMemory(DType.BFLOAT16, 1);
        assertThrows(IndexOutOfBoundsException.class, () -> array.get(1));
        assertThrows(IndexOutOfBoundsException.class, () -> array.set(-1, 1));
    }

    @Test
    void lazyTypedStorageDoesNotAllocateAFloatShadow() {
        var memory = new TensorMemory(DType.FLOAT16, 4, false);
        assertFalse(memory.hasHostStorage());
        assertThrows(IllegalStateException.class, memory::rawFloats);
        memory.set(1, 1.5f);
        assertTrue(memory.hasHostStorage());
        assertEquals(1.5f, memory.get(1));
        assertEquals(8, memory.byteLength());
        assertThrows(IllegalStateException.class, memory::floats);
        memory.discardStaleHost();
        assertFalse(memory.hasHostStorage());
    }

    @Test
    void copiesAreIndependentAndKeepRawCompactBits() {
        var memory = new TensorMemory(DType.BFLOAT16, 2);
        memory.shorts()[0] = (short) 0xff81;
        var copy = memory.copy();
        assertEquals(DType.BFLOAT16, copy.dtype());
        assertArrayEquals(memory.shorts(), copy.shorts());
        assertNotSame(memory.shorts(), copy.shorts());
        copy.shorts()[0] = 0;
        assertEquals((short) 0xff81, memory.shorts()[0]);
        memory.discardStaleHost();
        assertTrue(memory.hasHostStorage());
    }

    @Test
    void deviceRetentionsAreBalancedAcrossLogicalOwners() {
        var memory = new TensorMemory(DType.FLOAT32, 1, false);
        assertNull(memory.rawFloats());
        assertFalse(memory.deviceRetained());
        memory.retainDevice();
        memory.retainDevice();
        memory.releaseDevice();
        assertTrue(memory.deviceRetained());
        memory.releaseDevice();
        assertFalse(memory.deviceRetained());
        assertThrows(IllegalStateException.class, memory::releaseDevice);
    }
}
