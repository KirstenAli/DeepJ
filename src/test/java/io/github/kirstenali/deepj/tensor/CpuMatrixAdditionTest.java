package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuMatrixAdditionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @CsvSource({"63,65", "64,65", "67,131", "512,512", "513,515"})
    void transposesOffsetsAndTileTailsMatchLogicalAddition(int rows, int columns) {
        Tensor a = values(rows * columns + 7).slice(0, 3, rows * columns + 3).reshape(rows, columns);
        Tensor b = values(rows * columns + 9).slice(0, 5, rows * columns + 5)
                .reshape(columns, rows).transpose(0, 1);
        compare(a, b);
        compare(b, a);
        compare(b, b);
        Tensor target = values(rows * columns + 11).slice(0, 7, rows * columns + 7).reshape(rows, columns);
        float[] expected = sum(target, b);
        CpuAddition.into(target, b, target);
        assertArrayEquals(expected, target.toFloatArray());
    }

    @Test void sharedStorageSnapshotsBothInputLayoutsBeforeWriting() {
        Tensor input = values(67 * 67).reshape(67, 67);
        Tensor transpose = input.transpose(0, 1);
        float[] expected = sum(input, transpose);
        CpuAddition.into(input, transpose, input);
        assertArrayEquals(expected, input.toFloatArray());
        expected = sum(transpose, input);
        CpuAddition.into(transpose, input, input);
        assertArrayEquals(expected, input.toFloatArray());
    }

    @Test void paddedRowsAndUnsupportedColumnStepsRetainCorrectResults() {
        Tensor a = values(67 * 262).reshape(67, 262).slice(1, 0, 262, 2);
        Tensor b = values(131 * 67).reshape(131, 67).transpose(0, 1);
        compare(a, b);
        compare(values(134 * 131).reshape(134, 131).slice(0, 0, 134, 2), b);
        Tensor target = new Tensor(131, 67).transpose(0, 1);
        CpuAddition.into(b, b, target);
        assertArrayEquals(sum(b, b), target.toFloatArray());
    }

    private static void compare(Tensor a, Tensor b) {
        assertArrayEquals(sum(a, b), CpuAddition.apply(a, b).toFloatArray());
    }

    private static float[] sum(Tensor a, Tensor b) {
        float[] result = a.toFloatArray(), increment = b.toFloatArray();
        for (int index = 0; index < result.length; index++) { result[index] += increment[index]; }
        return result;
    }

    private static Tensor values(int count) {
        var random = new Random(count);
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = random.nextFloat() - .5f; }
        return Tensor.from(values, count);
    }
}
