package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class MetalHalfMatrixRoutingTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "FLOAT32"})
    void smallMatricesPreserveTheStridedKernelAccumulationOrder(DType leftType) {
        assertSameProduct(leftType, 8, 8, 8);
        assertSameProduct(leftType, 8, 576, 576);
        assertSameProduct(leftType, 15, 576, 576);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "FLOAT32"})
    void largeAndPartialTilesMatchTheExistingStridedKernel(DType leftType) {
        assertSameProduct(leftType, 64, 64, 64);
        assertSameProduct(leftType, 65, 256, 257);
    }

    @Test
    void largeMixedProductsNeverNarrowTheFloat32Operand() {
        float[] values = new float[64 * 256];
        for (int i = 0; i < values.length; i++) {
            values[i] = i % 2 == 0 ? 131072 : 1 + Math.scalb(1.0f, -20);
        }
        float[] identity = new float[256 * 256];
        for (int i = 0; i < 256; i++) { identity[i * 257] = 1; }
        Tensor left = Tensor.from(values, 64, 256);
        Tensor right = Tensor.from(identity, DType.FLOAT16, 256, 256);
        Tensor result = left.matmul(right);
        assertEquals(DType.FLOAT32, result.dtype());
        assertArrayEquals(values, result.toFloatArray());
    }

    @Test
    void largeHalfProductsAccumulateBeforeRounding() {
        float[] values = new float[64 * 64];
        for (int row = 0; row < 64; row++) {
            values[row * 64] = 4096;
            values[row * 64 + 1] = 1;
            values[row * 64 + 2] = -4096;
        }
        float[] ones = new float[values.length];
        Arrays.fill(ones, 1);
        Tensor left = Tensor.from(values, DType.FLOAT16, 64, 64);
        Tensor right = Tensor.from(ones, DType.FLOAT16, 64, 64);
        assertArrayEquals(ones, left.matmul(right).toFloatArray());
    }

    @Test
    void largeHalfProductsPreserveSubnormalAccumulation() {
        float minimum = Math.scalb(1.0f, -24);
        float[] values = new float[64 * 64], weights = new float[values.length];
        for (int row = 0; row < 64; row++) {
            values[row * 64] = minimum;
            values[row * 64 + 1] = minimum;
        }
        Arrays.fill(weights, 0, 128, 0.5f);
        float[] expected = new float[values.length];
        Arrays.fill(expected, minimum);
        Tensor left = Tensor.from(values, DType.FLOAT16, 64, 64);
        Tensor right = Tensor.from(weights, DType.FLOAT16, 64, 64);
        assertArrayEquals(expected, left.matmul(right).toFloatArray());
    }

    @Test
    void largePartialHalfTilesPreserveInfinityAndNan() {
        float[] values = new float[65 * 64], ones = new float[64 * 65];
        Arrays.fill(ones, 1);
        Tensor right = Tensor.from(ones, DType.FLOAT16, 64, 65);
        for (float special : new float[]{Float.POSITIVE_INFINITY, Float.NaN}) {
            for (int row = 0; row < 65; row++) { values[row * 64] = special; }
            float[] expected = new float[65 * 65];
            Arrays.fill(expected, special);
            Tensor left = Tensor.from(values, DType.FLOAT16, 65, 64);
            assertArrayEquals(expected, left.matmul(right).toFloatArray());
        }
    }

    private static void assertSameProduct(DType type, int rows, int inner, int columns) {
        Tensor left = Tensor.from(values(rows, inner, 9091), type, rows, inner);
        float[] values = values(inner, columns, 5091), padded = new float[values.length * 2];
        for (int i = 0; i < values.length; i++) { padded[i * 2] = values[i]; }
        Tensor right = Tensor.from(values, DType.FLOAT16, inner, columns);
        Tensor strided = Tensor.from(padded, DType.FLOAT16, inner, columns * 2)
                .slice(1, 0, columns * 2, 2);
        assertArrayEquals(left.matmul(strided).toFloatArray(), left.matmul(right).toFloatArray());
    }

    private static float[] values(int rows, int columns, long seed) {
        Random random = new Random(seed + rows + columns);
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) { values[i] = (float) (random.nextGaussian() * .02); }
        return values;
    }
}
