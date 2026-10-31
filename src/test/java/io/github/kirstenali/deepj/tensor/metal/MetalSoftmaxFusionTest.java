package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalSoftmaxFusionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @CsvSource({"3,1,0.125", "6,7,0.25", "9,65,-0.7", "8,257,0.0", "12,1024,0.125"})
    void fusedScaleMatchesTheSeparateMetalOperationsExactly(int rows, int cols, float scale) {
        Tensor probabilities = metal.softmaxRows(random(rows, cols, 17));
        Tensor gradient = random(rows, cols, 29);
        Tensor expected = metal.multiplyScalar(metal.softmaxBackward(gradient, probabilities), scale);
        Tensor actual = metal.softmaxBackward(gradient, probabilities, scale);
        expected.materialize();
        actual.materialize();
        assertArrayEquals(expected.rawData(), actual.rawData());
    }

    @ParameterizedTest
    @CsvSource({"1,1", "3,5", "4,65", "2,1024"})
    void causalGradientsMatchCpuWithoutChangingInputs(int heads, int length) {
        Tensor scores = random(heads * length, length, 31);
        Tensor gradient = random(scores.matrixRows(), scores.matrixCols(), 41);
        Tensor original = new Tensor(gradient);
        float scale = 0.125f;
        Tensor expected = cpu.softmaxBackward(gradient, cpu.causalSoftmax(scores, length, scale), scale);
        Tensor actual = metal.softmaxBackward(gradient, metal.causalSoftmax(scores, length, scale), scale);
        assertTensorClose(expected, actual, 2e-6f, 2e-4f);
        gradient.materialize();
        assertArrayEquals(original.rawData(), gradient.rawData());
    }

    @Test
    void incompatibleGradientShapesFailBeforeRecordingWork() {
        assertThrows(IllegalArgumentException.class,
                () -> metal.softmaxBackward(Tensor.ones(2, 3), Tensor.ones(3, 2), 0.125f));
    }
}
