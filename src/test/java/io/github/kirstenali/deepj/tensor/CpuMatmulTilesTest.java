package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuMatmulTilesTest {
    @BeforeEach void backend() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void tileBoundariesAndAllPhysicalLayoutsPreserveScalarReferenceBits(DType dtype) {
        for (int rows : new int[]{1, 3, 4, 5, 9}) {
            Tensor left = values(rows + 2, 37, dtype).slice(0, 1, rows + 1).slice(1, 1, 36);
            check(left, values(35, 259, dtype));
            check(left, values(259, 35, dtype).transpose());
        }
    }

    @Test void allMixedStoragePairsAccumulateInFloat32AndRoundOnlyTheOutput() {
        for (DType first : DType.values()) {
            for (DType second : DType.values()) {
                check(values(5, 35, first), values(129, 35, second).transpose());
            }
        }
    }

    @Test void parallelismChangesNeitherTheResultNorTheInputStorage() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        Tensor left = values(9, 35, DType.FLOAT32), right = values(35, 259, DType.FLOAT32);
        float[] original = right.toFloatArray();
        try {
            DeepJExecutor.setParallelEnabled(false);
            float[] sequential = left.matmul(right).toFloatArray();
            DeepJExecutor.setParallelEnabled(true);
            assertArrayEquals(bits(sequential), bits(left.matmul(right).toFloatArray()));
            assertArrayEquals(original, right.toFloatArray());
        } finally { DeepJExecutor.setParallelEnabled(previous); }
    }

    @Test void subsequentCallsReadUpdatedWeightsRatherThanAStaleConvertedCopy() {
        Tensor left = values(1, 35, DType.BFLOAT16), weight = values(259, 35, DType.BFLOAT16);
        check(left, weight.transpose());
        weight.set(128, 17, 3);
        check(left, weight.transpose());
        weight.set(258, 34, -2);
        check(left, weight.transpose());
    }

    @Test void backwardAcrossBothTileBoundariesMatchesIndependentScalarGradients() {
        Tensor left = values(5, 35, DType.FLOAT32).requiresGrad(true);
        Tensor right = values(35, 129, DType.FLOAT32).requiresGrad(true);
        Tensor output = left.matmul(right);
        float[] expected = product(left, right);
        output.multiply(output).sumTensor().backward();
        assertArrayEquals(leftGradient(left, right, expected), left.grad().toFloatArray(), 2e-5f);
        assertArrayEquals(rightGradient(left, right, expected), right.grad().toFloatArray(), 2e-5f);
    }

    private static void check(Tensor left, Tensor right) {
        float[] a = left.toFloatArray(), b = right.toFloatArray();
        Tensor expected = Tensor.from(product(left, right), DType.promote(left.dtype(), right.dtype()),
                left.size(0), right.size(1));
        Tensor actual = left.matmul(right);
        assertEquals(expected.dtype(), actual.dtype());
        assertArrayEquals(expected.shape(), actual.shape());
        assertArrayEquals(bits(expected.toFloatArray()), bits(actual.toFloatArray()));
        assertArrayEquals(a, left.toFloatArray());
        assertArrayEquals(b, right.toFloatArray());
    }

    private static float[] product(Tensor left, Tensor right) {
        int rows = left.size(0), inner = left.size(1), columns = right.size(1);
        float[] a = left.toFloatArray(), b = right.toFloatArray(), output = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            referenceRow(a, b, output, row, inner, columns);
        }
        return output;
    }

    private static void referenceRow(float[] a, float[] b, float[] output, int row, int inner, int columns) {
        for (int column = 0; column < columns; column++) {
            float sum = 0;
            for (int k = 0; k < inner; k++) { sum += a[row * inner + k] * b[k * columns + column]; }
            output[row * columns + column] = sum;
        }
    }

    private static float[] leftGradient(Tensor a, Tensor b, float[] output) {
        float[] right = b.toFloatArray(), result = new float[a.numel()];
        for (int row = 0; row < a.size(0); row++) {
            for (int k = 0; k < a.size(1); k++) {
                result[row * a.size(1) + k] = leftDerivative(right, output, row, k, b.size(1));
            }
        }
        return result;
    }

    private static float leftDerivative(float[] weight, float[] output, int row, int k, int columns) {
        float sum = 0;
        for (int column = 0; column < columns; column++) {
            sum += 2 * output[row * columns + column] * weight[k * columns + column];
        }
        return sum;
    }

    private static float[] rightGradient(Tensor a, Tensor b, float[] output) {
        float[] input = a.toFloatArray(), result = new float[b.numel()];
        for (int k = 0; k < b.size(0); k++) {
            for (int column = 0; column < b.size(1); column++) {
                result[k * b.size(1) + column] =
                        rightDerivative(input, output, a.size(0), a.size(1), b.size(1), k, column);
            }
        }
        return result;
    }

    private static float rightDerivative(float[] a, float[] output, int rows, int inner,
                                          int columns, int k, int c) {
        float sum = 0;
        for (int row = 0; row < rows; row++) { sum += a[row * inner + k] * (2 * output[row * columns + c]); }
        return sum;
    }

    private static Tensor values(int rows, int columns, DType dtype) {
        var random = new Random(37L * rows + columns);
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * .25f - .125f; }
        return Tensor.from(values, dtype, rows, columns);
    }

    private static int[] bits(float[] values) {
        int[] output = new int[values.length];
        for (int i = 0; i < values.length; i++) { output[i] = Float.floatToIntBits(values[i]); }
        return output;
    }
}
