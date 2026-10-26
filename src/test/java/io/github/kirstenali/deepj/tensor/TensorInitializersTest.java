package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorInitializersTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void initializationWritesOnlySelectedStorageAndPreservesGaussianDrawOrder(DType dtype) {
        Random random = new Random(91), reference = new Random(91);
        Tensor actual = Tensor.random(dtype, random, 2, 3, 4);
        for (int index = 0; index < actual.numel(); index++) {
            float expected = quantize((float) (reference.nextGaussian() * 0.1), dtype);
            assertEquals(expected, actual.reshape(24).get(index));
        }
        assertEquals(reference.nextLong(), random.nextLong());
        assertEquals(dtype.storageBytes(24), actual.storageBytes());
        assertEquals(0, actual.gradientState().version());
        assertFalse(actual.requiresGrad());
        if (dtype != DType.FLOAT32) {
            assertThrows(IllegalStateException.class, actual::rawData);
            assertEquals(24, actual.memory().shorts().length);
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void constantsHandleScalarsAndEmptyAxesWithoutRandomDraws(DType dtype) {
        assertEquals(1.0f, Tensor.ones(dtype).item());
        assertEquals(0.0f, Tensor.zeros(dtype).item());
        assertArrayEquals(new float[]{1, 1, 1, 1}, Tensor.ones(dtype, 2, 2).toFloatArray());
        Random empty = new Random(77), reference = new Random(77);
        assertEquals(0, Tensor.random(dtype, empty, 2, 0, 3).numel());
        assertEquals(reference.nextLong(), empty.nextLong());
        assertEquals(0, Tensor.ones(dtype, 0, 4).storageBytes());
    }

    @Test
    void defaultMatrixInitializerHasIdenticalBitsAndInvalidInputsAreRejected() {
        assertArrayEquals(Tensor.random(3, 4, new Random(8)).toFloatArray(),
                Tensor.random(DType.FLOAT32, new Random(8), 3, 4).toFloatArray());
        assertThrows(NullPointerException.class, () -> Tensor.random(DType.FLOAT16, null, 2));
        assertThrows(NullPointerException.class, () -> Tensor.ones(null, 2));
        assertThrows(IllegalArgumentException.class, () -> Tensor.zeros(DType.FLOAT16, -1));
    }

    private static float quantize(float value, DType dtype) {
        if (dtype == DType.FLOAT16) {
            return CompactFloatReference.half(Short.toUnsignedInt(Float.floatToFloat16(value)));
        }
        if (dtype == DType.BFLOAT16) return Float.intBitsToFloat(CompactFloatReference.bfloat(value) << 16);
        return value;
    }
}
