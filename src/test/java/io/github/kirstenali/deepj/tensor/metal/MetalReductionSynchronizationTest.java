package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalReductionSynchronizationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @ValueSource(ints = {768, 769, 1536, 1664, 2112})
    void reusedScratchPreservesTheMeanBeforeComputingVariance(int columns) {
        Tensor input = constant(128, columns, 1);
        for (int round = 0; round < 8; round++) {
            Tensor variance = metal.varianceAlongRows(input);
            variance.materialize();
            for (float value : variance.data) {
                assertEquals(0.0f, value, 1e-12f);
            }
            metal.releaseTemporaryResources();
        }
    }

    @ParameterizedTest
    @ValueSource(floats = {-100, 0, 100})
    void fusedCrossEntropyPreservesTheMaximumAcrossSimdGroups(float shift) {
        int rows = 64, columns = 16_384;
        Tensor logits = constant(rows, columns, shift);
        for (int round = 0; round < 8; round++) {
            CrossEntropyResult result = metal.crossEntropy(logits, new int[rows]);
            assertEquals(Math.log(columns), result.meanLoss(), 4e-6);
            assertUniformGradient(result.gradient(), rows, columns);
            metal.releaseTemporaryResources();
        }
    }

    @ParameterizedTest
    @ValueSource(floats = {-100, 0, 100})
    void separateCrossEntropyPreservesTheMaximumAcrossSimdGroups(float shift) {
        int rows = 64, columns = 16_384;
        Tensor logits = constant(rows, columns, shift);
        int[] targets = new int[rows];
        for (int round = 0; round < 8; round++) {
            assertEquals(Math.log(columns), metal.crossEntropyLoss(logits, targets), 4e-6);
            assertUniformGradient(metal.crossEntropyGradient(logits, targets), rows, columns);
            metal.releaseTemporaryResources();
        }
    }

    @ParameterizedTest
    @ValueSource(floats = {-100, 0, 100})
    void causalSoftmaxPreservesTheMaximumAcrossSimdGroups(float shift) {
        int length = 1024;
        Tensor scores = constant(length, length, shift);
        for (int round = 0; round < 8; round++) {
            Tensor probabilities = metal.causalSoftmax(scores, length, 0.125f);
            assertUniformCausalRows(probabilities);
            metal.releaseTemporaryResources();
        }
    }

    private static Tensor constant(int rows, int columns, float value) {
        Tensor tensor = new Tensor(rows, columns);
        Arrays.fill(tensor.data, value);
        return tensor;
    }

    private static void assertUniformGradient(Tensor gradient, int rows, int columns) {
        gradient.materialize();
        float other = 1.0f / (rows * columns);
        float target = (1.0f / columns - 1) / rows;
        for (int index = 0; index < gradient.data.length; index++) {
            assertEquals(index % columns == 0 ? target : other, gradient.data[index]);
        }
    }

    private static void assertUniformCausalRows(Tensor probabilities) {
        probabilities.materialize();
        for (int index = 0; index < probabilities.data.length; index++) {
            int row = index / probabilities.cols, column = index % probabilities.cols;
            float expected = column <= row ? 1.0f / (row + 1) : 0;
            assertEquals(expected, probabilities.data[index], 2e-7f);
        }
    }
}
