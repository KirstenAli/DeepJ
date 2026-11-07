package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuAxisSoftmaxTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test void denseHigherRanksMatchIndependentForwardAndBackwardRows() {
        for (int[] shape : new int[][]{{17}, {3, 17}, {2, 3, 5, 17}}) {
            Tensor input = values(shape), gradient = values(shape);
            float[] probabilities = probabilities(input.toFloatArray(), 17);
            Tensor output = Tensor.backend().softmaxAxis(input, -1);
            assertArrayEquals(probabilities, output.toFloatArray());
            assertArrayEquals(derivatives(probabilities, gradient.toFloatArray(), 17),
                    Tensor.backend().softmaxAxisBackward(gradient, output, -1).toFloatArray());
        }
    }

    @Test void offsetAndPermutedInputsRetainTheirValues() {
        Tensor input = values(5, 3, 17).slice(0, 1, 4).permute(1, 0, 2);
        assertArrayEquals(probabilities(input.toFloatArray(), 17), input.softmax(-1).toFloatArray());
        assertArrayEquals(new float[0], new Tensor(2, 0).softmax(-1).toFloatArray());
    }

    private static float[] probabilities(float[] values, int width) {
        for (int start = 0; start < values.length; start += width) {
            float maximum = Float.NEGATIVE_INFINITY, sum = 0;
            for (int column = 0; column < width; column++) { maximum = Math.max(maximum, values[start + column]); }
            for (int column = 0; column < width; column++) {
                values[start + column] = (float) Math.exp(values[start + column] - maximum);
                sum += values[start + column];
            }
            for (int column = 0; column < width; column++) { values[start + column] /= sum; }
        }
        return values;
    }

    private static float[] derivatives(float[] probabilities, float[] gradient, int width) {
        for (int start = 0; start < gradient.length; start += width) {
            float dot = 0;
            for (int column = 0; column < width; column++) {
                dot += gradient[start + column] * probabilities[start + column];
            }
            for (int column = 0; column < width; column++) {
                gradient[start + column] = probabilities[start + column] * (gradient[start + column] - dot);
            }
        }
        return gradient;
    }

    private static Tensor values(int... shape) {
        Tensor result = new Tensor(shape);
        Random random = new Random(17);
        for (int index = 0; index < result.numel(); index++) { result.rawData()[index] = random.nextFloat() * 20 - 10; }
        return result;
    }
}
