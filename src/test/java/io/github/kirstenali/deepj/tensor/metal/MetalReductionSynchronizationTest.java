package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalReductionSynchronizationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @ValueSource(ints = {768, 769, 1536, 1664, 2112})
    void reusedScratchPreservesTheMeanBeforeComputingVariance(int columns) {
        assertConstantVariance(128, columns, 8);
    }

    @Test
    void repeatedDispatchesExposeRareScratchReuseRaces() {
        assertConstantVariance(2048, 768, 1000);
    }

    private void assertConstantVariance(int rows, int columns, int rounds) {
        Tensor input = Tensor.ones(rows, columns);
        for (int round = 0; round < rounds; round++) {
            Tensor variance = metal.varianceAlongRows(input);
            variance.materialize();
            for (float value : variance.rawData()) {
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
        assertConstantCausalSoftmax(shift, 8);
    }

    @Test
    void repeatedCausalSoftmaxPreservesUniformRows() {
        assertConstantCausalSoftmax(100, 128);
    }

    private void assertConstantCausalSoftmax(float shift, int rounds) {
        int length = 1024;
        Tensor scores = constant(length, length, shift);
        for (int round = 0; round < rounds; round++) {
            Tensor probabilities = metal.causalSoftmax(scores, length, 0.125f);
            assertUniformCausalRows(probabilities);
            metal.releaseTemporaryResources();
        }
    }

    private static Tensor constant(int rows, int columns, float value) {
        Tensor tensor = new Tensor(rows, columns);
        Arrays.fill(tensor.rawData(), value);
        return tensor;
    }

    private static void assertUniformGradient(Tensor gradient, int rows, int columns) {
        gradient.materialize();
        float other = 1.0f / (rows * columns);
        float target = (1.0f / columns - 1) / rows;
        for (int index = 0; index < gradient.rawData().length; index++) {
            assertEquals(index % columns == 0 ? target : other, gradient.rawData()[index]);
        }
    }

    private static void assertUniformCausalRows(Tensor probabilities) {
        probabilities.materialize();
        for (int index = 0; index < probabilities.rawData().length; index++) {
            int row = index / probabilities.cols, column = index % probabilities.cols;
            float expected = column <= row ? 1.0f / (row + 1) : 0;
            assertEquals(expected, probabilities.rawData()[index], 2e-7f);
        }
    }
}
