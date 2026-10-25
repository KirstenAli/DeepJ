package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalScatterRepeatabilityTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @ValueSource(ints = {17, 1024})
    void duplicateRowsAccumulateInInputOrder(int count) {
        Tensor initial = random(7, 33, 42);
        Tensor gradient = cancellingGradients(count, initial.cols).retainDeviceBuffer();
        int[] indices = repeatedIndices(count, initial.rows);
        Tensor expected = new Tensor(initial);
        cpu.scatterAddRows(expected, indices, gradient);
        for (int round = 0; round < 64; round++) {
            Tensor actual = new Tensor(initial);
            metal.scatterAddRows(actual, indices, gradient);
            actual.materialize();
            assertArrayEquals(expected.rawData(), actual.rawData(), "round " + round);
            metal.releaseTemporaryResources();
        }
    }

    private static Tensor cancellingGradients(int rows, int columns) {
        Tensor result = new Tensor(rows, columns);
        float[] magnitudes = {1e6f, 0.01f, -1e6f, 0.25f};
        for (int index = 0; index < result.rawData().length; index++) {
            result.rawData()[index] = magnitudes[index % magnitudes.length];
        }
        return result;
    }

    private static int[] repeatedIndices(int count, int rows) {
        int[] result = new int[count];
        for (int index = 0; index < count; index++) {
            result[index] = index % rows;
        }
        return result;
    }
}
