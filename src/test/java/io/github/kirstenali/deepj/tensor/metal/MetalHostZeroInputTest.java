package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalHostZeroInputTest {
    @Test
    void onlyCpuZerosWithAGpuSourceQualify() {
        Tensor target = Tensor.zeros(1, 4), source = Tensor.ones(1, 4);
        assertFalse(MetalHostZeroInput.canSkip(target, source));
        source.setGpuTag(new GpuBuffer(0, 1, 4, false));
        assertTrue(MetalHostZeroInput.canSkip(target, source));
        target.setGpuTag(new GpuBuffer(1, 1, 4, false));
        assertFalse(MetalHostZeroInput.canSkip(target, source));
        target.setGpuTag(null);
        target.data = null;
        assertFalse(MetalHostZeroInput.canSkip(target, source));
        target.data = new float[1];
        assertFalse(MetalHostZeroInput.canSkip(target, source));
    }

    @Test
    void checksEveryChunkAndRejectsNegativeZeroAndAllOtherNonzeroBits() {
        float[] zeros = new float[32_781];
        assertTrue(MetalHostZeroInput.positiveZeros(zeros));
        for (int index : new int[]{0, 16_383, 16_384, zeros.length - 1}) {
            for (float value : new float[]{-0.0f, 1, -1, Float.MIN_VALUE, -Float.MIN_VALUE,
                    Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN,
                    Float.intBitsToFloat(0x7f800001)}) {
                zeros[index] = value;
                assertFalse(MetalHostZeroInput.positiveZeros(zeros));
                zeros[index] = 0;
            }
        }
        assertTrue(MetalHostZeroInput.positiveZeros(zeros));
        assertTrue(MetalHostZeroInput.positiveZeros(new float[0]));
    }
}
