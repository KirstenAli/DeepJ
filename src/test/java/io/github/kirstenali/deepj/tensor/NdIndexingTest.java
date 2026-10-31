package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NdIndexingTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void arbitraryAxisSelectionHandlesRepeatedIndicesAndStridedInput() {
        Tensor input = Tensor.from(new float[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 2, 3, 2);
        assertArrayEquals(new float[]{4, 5, 0, 1, 4, 5, 10, 11, 6, 7, 10, 11},
                input.indexSelect(1, 2, 0, 2).toFloatArray());
        assertArrayEquals(new float[]{4, 10, 0, 6, 4, 10, 5, 11, 1, 7, 5, 11},
                input.permute(2, 1, 0).indexSelect(1, 2, 0, 2).toFloatArray());
        assertArrayEquals(new float[]{30, 30, 10}, Tensor.from(new float[]{10, 20, 30})
                .indexSelect(-1, 2, 2, 0).toFloatArray());
        assertArrayEquals(new int[]{2, 0, 2}, input.indexSelect(1).shape());
    }

    @Test
    void indexSelectionPreservesRawCompactStorageBits() {
        Tensor input = new Tensor(DType.FLOAT16, 3);
        short[] values = {(short) 0x7c01, (short) 0x8000, (short) 0xff81};
        System.arraycopy(values, 0, input.memory().shorts(), 0, values.length);
        Tensor output = input.indexSelect(0, 2, 0, 1);
        assertArrayEquals(new short[]{values[2], values[0], values[1]}, output.memory().shorts());
        assertEquals(6, output.storageBytes());
    }

    @Test
    void repeatedIndexedUpdatesAccumulateWithoutCrossingOuterExamples() {
        Tensor target = new Tensor(2, 3, 2);
        Tensor updates = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 3, 2);
        target.indexAddInPlace(1, new int[]{2, 0, 2}, updates);
        assertArrayEquals(new float[]{3, 4, 0, 0, 6, 8, 9, 10, 0, 0, 18, 20}, target.toFloatArray());
        Tensor alias = Tensor.from(new float[]{1, 2});
        alias.indexAddInPlace(0, new int[]{1, 0}, alias);
        assertArrayEquals(new float[]{3, 3}, alias.toFloatArray());
    }

    @Test
    void compactIndexedUpdatesAccumulateInFloat32AndRoundOnce() {
        Tensor target = Tensor.from(new float[]{2048}, DType.FLOAT16);
        target.indexAddInPlace(0, new int[]{0, 0}, Tensor.from(new float[]{1, 1}, DType.FLOAT16));
        assertEquals(2050, target.get(0));
        assertEquals(DType.FLOAT16, target.dtype());
        assertEquals(2, target.storageBytes());
        Tensor empty = new Tensor(0, 3);
        empty.indexAddInPlace(1, new int[]{1}, new Tensor(0, 1));
        assertEquals(0, empty.numel());
    }

    @Test
    void invalidUpdatesDoNotChangeDataOrMutationVersion() {
        Tensor target = new Tensor(2, 3);
        assertThrows(IndexOutOfBoundsException.class, () -> target.indexSelect(0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> target.indexSelect(0, 2));
        assertThrows(IllegalArgumentException.class, () -> target.indexAddInPlace(0, new int[]{1}, new Tensor(3)));
        assertEquals(0, target.gradientState().version());
        Tensor compact = new Tensor(DType.FLOAT16, 2);
        assertThrows(IllegalArgumentException.class, () -> compact.indexAddInPlace(0, new int[]{1}, new Tensor(1)));
        assertEquals(0, compact.gradientState().version());
        Tensor expanded = Tensor.from(new float[]{1}).expand(2);
        assertThrows(IllegalStateException.class, () -> expanded.indexAddInPlace(0, new int[]{1}, new Tensor(1)));
    }
}
