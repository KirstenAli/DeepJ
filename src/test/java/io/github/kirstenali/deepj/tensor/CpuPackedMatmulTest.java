package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class CpuPackedMatmulTest {
    @BeforeEach void backend() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void allTransposeAndBatchCombinationsPreserveIndependentReferenceBits() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            for (boolean parallel : new boolean[]{false, true}) {
                DeepJExecutor.setParallelEnabled(parallel);
                checkCombinations();
            }
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private static void checkCombinations() {
        for (int batches : new int[]{1, 2, 3}) {
            for (int flags = 0; flags < 4; flags++) {
                check(batches, 4, 5, 7, (flags & 1) != 0, (flags & 2) != 0);
                check(batches, 1, 3, 0, (flags & 1) != 0, (flags & 2) != 0);
                check(batches, 0, 5, 7, (flags & 1) != 0, (flags & 2) != 0);
                check(batches, 4, 0, 7, (flags & 1) != 0, (flags & 2) != 0);
            }
        }
    }

    @Test
    void specialValuesKeepScalarOrderWithoutChangingInputs() {
        Tensor left = Tensor.from(new float[]{-0.0f, Float.MIN_VALUE, 1e30f, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.NaN, 1, -1}, 4, 2);
        Tensor right = Tensor.from(new float[]{1, -0.0f, 0, 1}, 4, 1);
        int[] leftBits = bits(left.toFloatArray()), rightBits = bits(right.toFloatArray());
        float[] expected = reference(left, right, 2, 2, 1, 2, false, false);
        float[] actual = new CpuBackend().batchedMatmul(left, right, 2, false, false).toFloatArray();
        for (int index = 0; index < expected.length; index++) {
            if (Float.isNaN(expected[index])) assertTrue(Float.isNaN(actual[index]));
            else assertEquals(Float.floatToRawIntBits(expected[index]), Float.floatToRawIntBits(actual[index]));
        }
        assertArrayEquals(leftBits, bits(left.toFloatArray()));
        assertArrayEquals(rightBits, bits(right.toFloatArray()));
    }

    private static void check(int batches, int rows, int columns, int inner, boolean ta, boolean tb) {
        Tensor left = input(batches * (ta ? inner : rows), ta ? rows : inner, 1);
        Tensor right = input(batches * (tb ? columns : inner), tb ? inner : columns, 2);
        Tensor actual = new CpuBackend().batchedMatmul(left, right, batches, ta, tb);
        float[] expected = reference(left, right, batches, rows, columns, inner, ta, tb);
        assertArrayEquals(new int[]{batches * rows, columns}, actual.shape());
        assertArrayEquals(bits(expected), bits(actual.toFloatArray()));
        Tensor direct = TensorBackendDefaults.batchedMatmul(left, right, batches, ta, tb);
        assertArrayEquals(bits(expected), bits(direct.toFloatArray()));
    }

    private static float[] reference(Tensor left, Tensor right, int batches, int rows, int columns,
                                     int inner, boolean ta, boolean tb) {
        float[] output = new float[batches * rows * columns];
        for (int batch = 0; batch < batches; batch++) {
            for (int row = 0; row < rows; row++) {
                referenceRow(left, right, output, batch, row, rows, columns, inner, ta, tb);
            }
        }
        return output;
    }

    private static void referenceRow(Tensor left, Tensor right, float[] output, int batch, int row,
                                     int rows, int columns, int inner, boolean ta, boolean tb) {
        for (int column = 0; column < columns; column++) {
            float sum = 0;
            for (int k = 0; k < inner; k++) {
                float a = ta ? left.get(batch * inner + k, row) : left.get(batch * rows + row, k);
                float b = tb ? right.get(batch * columns + column, k) : right.get(batch * inner + k, column);
                sum += a * b;
            }
            output[(batch * rows + row) * columns + column] = sum;
        }
    }

    private static Tensor input(int rows, int columns, int seed) {
        var random = new Random(seed);
        float[] values = new float[rows * columns];
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextFloat() * 4 - 2;
        }
        return Tensor.from(values, rows, columns);
    }

    private static int[] bits(float[] values) {
        int[] output = new int[values.length];
        for (int index = 0; index < values.length; index++) {
            output[index] = Float.floatToRawIntBits(values[index]);
        }
        return output;
    }
}
