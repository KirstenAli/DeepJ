package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuHeadPrefixProductTest {
    @ParameterizedTest @EnumSource(DType.class)
    void groupedQueriesAndOffsetCachesMatchScalarProductsWithoutReadingUnusedCapacity(DType dtype) {
        Tensor.setBackend(new CpuBackend());
        Tensor cache = values(dtype, 16, 19).slice(0, 1, 15).slice(1, 1, 18);
        for (int groups : new int[]{1, 2, 4, 5}) {
            check(values(dtype, 2 * groups, 17), cache, groups, true);
            check(values(dtype, 2 * groups, 3), cache, groups, false);
        }
    }

    private static void check(Tensor left, Tensor cache, int groups, boolean transpose) {
        float[] original = cache.toFloatArray();
        int columns = transpose ? 3 : 17;
        float[] expected = new float[2 * groups * columns];
        for (int row = 0; row < 2 * groups; row++) {
            for (int col = 0; col < columns; col++) {
                expected[row * columns + col] = dot(left, cache, row, col, groups, transpose);
            }
        }
        Tensor rounded = Tensor.from(expected, left.dtype(), 2 * groups, columns);
        Tensor actual = Tensor.backend().headPrefixMatmul(left, cache, 2, 3, transpose);
        assertArrayEquals(rounded.toFloatArray(), actual.toFloatArray());
        assertArrayEquals(original, cache.toFloatArray());
    }

    private static float dot(Tensor left, Tensor cache, int row, int column, int groups, boolean transpose) {
        int start = row / groups * 7, inner = transpose ? 17 : 3;
        float sum = 0;
        for (int k = 0; k < inner; k++) {
            sum += left.get(row, k) * (transpose ? cache.get(start + column, k) : cache.get(start + k, column));
        }
        return sum;
    }

    private static Tensor values(DType dtype, int rows, int columns) {
        var random = new Random(37L * rows + columns);
        float[] values = new float[rows * columns];
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() - .5f; }
        return Tensor.from(values, dtype, rows, columns);
    }
}
