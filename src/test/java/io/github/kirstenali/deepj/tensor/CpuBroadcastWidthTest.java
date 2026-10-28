package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class CpuBroadcastWidthTest {
    @BeforeEach
    void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void broadcastsMatchIndependentScalarBitsIncludingSignedZeros(DType dtype) {
        Tensor input = Tensor.from(new float[]{-0.0f, 0, -1, .25f, 2, 0, -0.0f, 3, -.5f, 1.25f}, dtype, 2, 5);
        Tensor row = Tensor.from(new float[]{-0.0f, 0, .5f, -.25f, 0}, dtype, 1, 5);
        Tensor column = Tensor.from(new float[]{-2, 0}, dtype, 2, 1);
        check(input, row, column);
    }

    @Test
    void zeroRowsAndColumnsRemainValidEmptyMatrixResults() {
        for (int[] shape : new int[][]{{0, 5}, {3, 0}}) {
            Tensor input = new Tensor(shape);
            check(input, new Tensor(1, shape[1]), new Tensor(shape[0], 1));
        }
    }

    @Test
    void offsetAndPermutedInputsRetainLogicalFallbackValues() {
        Tensor base = Tensor.from(new float[]{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12}, 4, 3);
        Tensor offset = base.slice(0, 1, 3);
        Tensor permuted = Tensor.from(new float[]{1, -2, 3, -4, 5, -6}, 3, 2).transpose(0, 1);
        for (Tensor input : new Tensor[]{offset, permuted}) {
            check(input, Tensor.from(new float[]{.5f, -1, 2}, 1, 3), Tensor.from(new float[]{-2, 3}, 2, 1));
        }
    }

    @Test
    void incorrectBroadcastVectorsAreStillRejected() {
        Tensor input = new Tensor(2, 3);
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().addBroadcastRows(input, new Tensor(2, 3)));
        assertThrows(IllegalArgumentException.class,
                () -> Tensor.backend().multiplyBroadcastCols(input, new Tensor(1, 1)));
    }

    private static void check(Tensor input, Tensor row, Tensor column) {
        Tensor sum = Tensor.backend().addBroadcastRows(input, row);
        Tensor product = Tensor.backend().multiplyBroadcastCols(input, column);
        assertArrayEquals(input.shape(), sum.shape());
        assertArrayEquals(input.shape(), product.shape());
        assertEquals(input.dtype(), sum.dtype());
        assertEquals(input.dtype(), product.dtype());
        for (int r = 0; r < input.matrixRows(); r++) {
            for (int c = 0; c < input.matrixCols(); c++) {
                assertBits(input.dtype(), input.get(r, c) + row.get(0, c), sum.get(r, c));
                assertBits(input.dtype(), input.get(r, c) * column.get(r, 0), product.get(r, c));
            }
        }
    }

    private static void assertBits(DType dtype, float expected, float actual) {
        float rounded = dtype == DType.FLOAT32 ? expected : dtype.decode16(dtype.encode16(expected));
        assertEquals(Float.floatToRawIntBits(rounded), Float.floatToRawIntBits(actual));
    }
}
