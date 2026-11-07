package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuStridedElementwiseTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test void transposedGradientAccumulationCrossesWorkerAndRowBoundaries() {
        Tensor source = values(1025, 17).transpose(0, 1);
        Tensor target = values(17, 1025);
        float[] expected = target.toFloatArray(), increment = source.toFloatArray();
        for (int index = 0; index < expected.length; index++) { expected[index] += increment[index]; }
        Tensor.backend().addInPlace(target, source);
        assertArrayEquals(expected, target.toFloatArray());
    }

    @Test void broadcastsAndPermutedOutputsPreserveLogicalOrder() {
        Tensor input = values(2, 17, 1025).permute(1, 0, 2);
        Tensor row = values(1025), output = new Tensor(2, 17, 1025).permute(1, 0, 2);
        float[] expected = input.toFloatArray(), bias = row.toFloatArray();
        for (int index = 0; index < expected.length; index++) { expected[index] -= bias[index % bias.length]; }
        CpuNdElementwise.binaryInto(input, row, output, (left, right) -> left - right);
        assertArrayEquals(expected, output.toFloatArray());
    }

    @Test void overlappingTransposeReadsTheOriginalValues() {
        Tensor input = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3);
        Tensor.backend().addInPlace(input, input.transpose(0, 1));
        assertArrayEquals(new float[]{2, 6, 10, 6, 10, 14, 10, 14, 18}, input.toFloatArray());
    }

    @Test void emptyAndScalarBroadcastsRemainValid() {
        Tensor input = Tensor.from(new float[]{-0.0f, 2, -3, Float.NaN}, 2, 2).transpose(0, 1);
        Tensor zero = Tensor.from(new float[]{0}, new int[0]);
        assertArrayEquals(new float[]{0, -3, 2, Float.NaN}, input.add(zero).toFloatArray());
        assertArrayEquals(new float[0], new Tensor(2, 0, 3).add(zero).toFloatArray());
    }

    private static Tensor values(int... shape) {
        Tensor result = new Tensor(shape);
        Random random = new Random(42);
        for (int index = 0; index < result.numel(); index++) { result.rawData()[index] = random.nextFloat() - .5f; }
        return result;
    }
}
