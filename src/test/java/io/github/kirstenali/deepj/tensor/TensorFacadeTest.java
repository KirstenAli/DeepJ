package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class TensorFacadeTest {

    private TensorBackend previous;

    @BeforeEach
    void useCpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }

    @AfterEach
    void restoreBackend() {
        Tensor.setBackend(previous);
    }

    @Test
    void arithmeticAndColumnBroadcastsReturnExpectedValues() {
        Tensor input = Tensor.from2D(new float[][]{{2, 4}, {6, 8}});
        Tensor column = Tensor.from2D(new float[][]{{2}, {4}});
        assertValues(input.divide(input), 1, 1, 1, 1);
        assertValues(input.divideScalar(2), 1, 2, 3, 4);
        assertValues(input.addBroadcastCols(column), 4, 6, 10, 12);
        assertValues(input.multiplyBroadcastCols(column), 4, 8, 24, 32);
        assertValues(input, 2, 4, 6, 8);
    }

    @Test
    void unaryAndReductionFacadesUseCorrectDimensions() {
        Tensor input = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        assertValues(input.neg(), -1, -2, -3, -4);
        assertValues(input.exp().log(), 1, 2, 3, 4);
        assertValues(input.clamp(2, 3), 2, 2, 3, 3);
        assertValues(input.pow(2), 1, 4, 9, 16);
        assertValues(input.sumAlongRows(), 3, 7);
        assertValues(input.sumAlongCols(), 4, 6);
        assertValues(input.maxAlongRows(), 2, 4);
    }

    @Test
    void rowCopiesAndSamplingPreserveTheSource() {
        Tensor input = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        Tensor copy = Tensor.zeros(2, 2);
        copy.setRow(0, input, 1);
        copy.setRow(1, input, 0);
        assertValues(copy, 3, 4, 1, 2);
        assertValues(Tensor.sampleRows(input, 2, new Random(42)), 3, 4, 1, 2);
        assertValues(input, 1, 2, 3, 4);
        assertValues(Tensor.causalMask(2), 0, -1e9f, 0, 0);
    }

    private static void assertValues(Tensor tensor, float... expected) {
        tensor.materialize();
        assertArrayEquals(expected, tensor.data, 1e-5f);
    }
}
