package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuAdditionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @ValueSource(ints = {0, 1, 7, 16384, 262143, 262144, 262149, 1048576})
    void offsetsAndParallelBoundariesMatchScalarArithmetic(int count) {
        Tensor left = values(count + 3).slice(0, 1, count + 1);
        Tensor right = values(count + 5).slice(0, 2, count + 2);
        float[] expected = left.toFloatArray(), increment = right.toFloatArray();
        for (int index = 0; index < count; index++) { expected[index] += increment[index]; }
        assertArrayEquals(expected, Tensor.backend().add(left, right).toFloatArray());
        Tensor.backend().addInPlace(left, right);
        assertArrayEquals(expected, left.toFloatArray());
    }

    @Test void overlappingViewsReadOriginalInputsAndSelfAdditionRemainsValid() {
        Tensor storage = Tensor.from(new float[]{1, 2, 3, 4, 5}, 5);
        Tensor.backend().addInPlace(storage.slice(0, 1, 5), storage.slice(0, 0, 4));
        assertArrayEquals(new float[]{1, 3, 5, 7, 9}, storage.toFloatArray());
        Tensor.backend().addInPlace(storage, storage);
        assertArrayEquals(new float[]{2, 6, 10, 14, 18}, storage.toFloatArray());
    }

    @Test void broadcastingStridesAndCompactStorageKeepTheirFallback() {
        Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor row = Tensor.from(new float[]{2, 3, 4}, 1, 3);
        assertArrayEquals(new float[]{3, 5, 7, 6, 8, 10}, Tensor.backend().add(matrix, row).toFloatArray());
        Tensor transpose = matrix.transpose(0, 1);
        assertArrayEquals(new float[]{2, 8, 4, 10, 6, 12}, Tensor.backend().add(transpose, transpose).toFloatArray());
        Tensor compact = matrix.to(DType.BFLOAT16);
        assertArrayEquals(new float[]{2, 4, 6, 8, 10, 12}, Tensor.backend().add(compact, compact).toFloatArray());
    }

    private static Tensor values(int count) {
        var random = new Random(count);
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = random.nextFloat() * 8 - 4; }
        return Tensor.from(values, count);
    }
}
