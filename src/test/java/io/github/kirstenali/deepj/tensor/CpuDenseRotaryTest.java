package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuDenseRotaryTest {
    @ParameterizedTest @CsvSource({"8,8", "256,128"})
    void denseOffsetsMatchStridedReferenceInBothDirections(int rows, int width) {
        Tensor.setBackend(new CpuBackend());
        Tensor input = Tensor.random(rows + 2, width, new Random(19)).slice(0, 1, rows + 1);
        Tensor cosine = Tensor.random(10, width / 2, new Random(20)).slice(0, 1, 9);
        Tensor sine = Tensor.random(10, width / 2, new Random(21)).slice(0, 1, 9);
        for (boolean halfSplit : new boolean[]{false, true}) {
            compare(input, cosine, sine, halfSplit, false);
            compare(input, cosine, sine, halfSplit, true);
        }
    }

    private static void compare(Tensor input, Tensor cosine, Tensor sine, boolean halfSplit, boolean inverse) {
        Tensor strided = strided(input);
        Tensor expected = CpuNdPosition.rotary(strided, cosine, sine, 8, inverse, halfSplit);
        Tensor actual = CpuNdPosition.rotary(input, cosine, sine, 8, inverse, halfSplit);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertArrayEquals(input.toFloatArray(), strided.toFloatArray());
        boolean parallel = DeepJExecutor.isParallelEnabled();
        try {
            DeepJExecutor.setParallelEnabled(false);
            assertArrayEquals(actual.toFloatArray(), CpuNdPosition.rotary(input, cosine, sine, 8, inverse,
                    halfSplit).toFloatArray());
        } finally { DeepJExecutor.setParallelEnabled(parallel); }
    }

    private static Tensor strided(Tensor input) {
        float[] values = input.toFloatArray(), padded = new float[values.length * 2];
        for (int index = 0; index < values.length; index++) { padded[2 * index] = values[index]; }
        return Tensor.from(padded, input.matrixRows(), input.matrixCols() * 2).slice(1, 0, input.matrixCols() * 2, 2);
    }
}
