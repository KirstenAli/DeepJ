package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CpuLegacyLayoutTest {
    private final CpuBackend cpu = new CpuBackend();
    @BeforeEach void backend() { Tensor.setBackend(cpu); }

    @Test
    void legacyRowScatterUsesTypedOrderedAccumulationAndHandlesAliasedStridedUpdates() {
        Tensor target = Tensor.from(new float[]{2048, 2048}, DType.FLOAT16, 1, 2);
        cpu.scatterAddRows(target, new int[]{0, 0}, Tensor.from(new float[]{1, 1, 1, 1}, DType.FLOAT16, 2, 2));
        assertArrayEquals(new float[]{2050, 2050}, target.toFloatArray());
        Tensor parent = Tensor.from(new float[]{9, 1, 2, 8, 7, 3, 4, 6}, DType.BFLOAT16, 2, 4);
        Tensor view = parent.slice(1, 1, 3, 1);
        cpu.scatterAddRows(view, new int[]{1, 0}, view);
        assertArrayEquals(new float[]{9, 4, 6, 8, 7, 4, 6, 6}, parent.toFloatArray());
        assertThrows(IndexOutOfBoundsException.class,
                () -> cpu.scatterAddRows(view, new int[]{2}, new Tensor(DType.BFLOAT16, 1, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> cpu.scatterAddRows(view, new int[]{0}, new Tensor(1, 2)));
    }

    @Test
    void legacyLayerNormAdjointAcceptsTypedNdLayoutsWithoutApplyingGainTwice() {
        Tensor gradient = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, DType.FLOAT16, 2, 1, 3);
        Tensor normalized = Tensor.from(new float[]{-1, 0, 1, -1, 0, 1}, DType.FLOAT16, 2, 1, 3);
        Tensor std = Tensor.from(new float[]{2, 3}, 2, 1, 1);
        Tensor expected = cpu.normalizationBackward(gradient, normalized, std, Tensor.ones(1, 3), true);
        Tensor actual = cpu.layerNormBackward(gradient, normalized, std, 3);
        assertEquals(DType.FLOAT32, actual.dtype());
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertArrayEquals(actual.toFloatArray(), cpu.layerNormBackward(gradient.permute(1, 0, 2),
                normalized.permute(1, 0, 2), std.permute(1, 0, 2), 3).toFloatArray());
        assertThrows(IllegalArgumentException.class, () -> cpu.layerNormBackward(gradient, normalized, std, 2));
        assertThrows(IllegalArgumentException.class,
                () -> cpu.layerNormBackward(gradient, normalized, new Tensor(2, 1), 3));
    }

    @Test
    void rowMaximumHasTheSameEmptyNonfiniteAndSignedZeroContractForAllDtypes() {
        for (DType dtype : DType.values()) {
            assertThrows(IllegalArgumentException.class, () -> cpu.maxAlongRows(new Tensor(dtype, 2, 0)));
            assertArrayEquals(new int[]{0, 1}, cpu.maxAlongRows(new Tensor(dtype, 0, 2)).shape());
            Tensor input = Tensor.from(new float[]{0, -0.0f, 1, Float.NaN, -0.0f, -0.0f}, dtype, 3, 2);
            float[] values = cpu.maxAlongRows(input).toFloatArray();
            assertEquals(0, Float.floatToRawIntBits(values[0]));
            assertTrue(Float.isNaN(values[1]));
            assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(values[2]));
        }
    }
}
