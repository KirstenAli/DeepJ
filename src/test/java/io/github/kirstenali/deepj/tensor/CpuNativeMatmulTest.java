package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CpuNativeMatmulTest {
    @BeforeEach void backend() {
        Tensor.setBackend(new CpuBackend());
        assumeTrue(CpuBlas.AVAILABLE, "Optional Apple CPU library unavailable");
    }

    @ParameterizedTest @EnumSource(DType.class)
    void largeProductsMatchAnIndependentDoublePrecisionReference(DType dtype) {
        check(values(dtype, 3, 513), values(dtype, 513, 257));
        check(values(dtype, 1, 513), values(dtype, 1025, 513).transpose());
        check(values(dtype, 3, 513), values(dtype, 257, 513).transpose());
        check(values(dtype, 65, 65), values(dtype, 65, 65));
        check(values(dtype, 2, 257), values(dtype, 257, 513));
        check(values(dtype, 1, 513), values(dtype, 1026, 1026).slice(0, 0, 1026, 2).slice(1, 0, 1026, 2));
    }

    @Test void mixedStorageAndOffsetViewsUseTheSamePromotionRules() {
        for (DType a : DType.values()) {
            for (DType b : DType.values()) {
                check(values(a, 5, 515).slice(0, 1, 4).slice(1, 1, 514), values(b, 513, 257));
            }
        }
    }

    @Test void largeBackwardMatchesIndependentDerivatives() {
        Tensor a = values(DType.FLOAT32, 3, 257).requiresGrad(true);
        Tensor b = values(DType.FLOAT32, 257, 513).requiresGrad(true);
        Tensor result = a.matmul(b);
        result.sumTensor().backward();
        checkLeftGradient(a, b);
        checkRightGradient(a, b);
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactPrefillHandlesTileBoundariesBothWeightLayoutsAndRowTails(DType dtype) {
        for (int inner : new int[]{2047, 2048, 2049}) {
            Tensor left = values(DType.FLOAT32, 3, inner);
            check(left, values(dtype, 259, inner).transpose());
            check(left, values(dtype, inner, 259));
        }
        check(values(DType.FLOAT32, 65, 2048), values(dtype, 2048, 129));
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void directFloatInputsPreserveOffsetsAndStridedInputsKeepTheirPacking(DType dtype) {
        Tensor right = values(dtype, 259, 2048).transpose(), parent = values(DType.FLOAT32, 5, 2048);
        check(parent.slice(0, 1, 4), right);
        check(parent.slice(0, 2, 3), right);
        check(values(DType.FLOAT32, 2048, 3).transpose(), right);
        check(values(DType.FLOAT32, 6, 2048).slice(0, 0, 6, 2), right);
        check(values(DType.FLOAT32, 1, 4096).slice(1, 0, 4096, 2), right);
    }

    private static void check(Tensor a, Tensor b) {
        float[] left = a.toFloatArray(), right = b.toFloatArray();
        Tensor result = a.matmul(b);
        assertEquals(DType.promote(a.dtype(), b.dtype()), result.dtype());
        float[] actual = result.toFloatArray();
        double precision = precision(result.dtype());
        for (int row = 0; row < a.size(0); row++) {
            checkRow(a, b, left, right, actual, precision, row);
        }
        assertArrayEquals(left, a.toFloatArray());
        assertArrayEquals(right, b.toFloatArray());
    }

    private static double precision(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> 2e-5;
            case FLOAT16 -> .001;
            case BFLOAT16 -> .008;
        };
    }

    private static void checkRow(Tensor a, Tensor b, float[] left, float[] right, float[] actual,
                                  double precision, int row) {
        for (int column = 0; column < b.size(1); column++) {
            double expected = dot(left, right, row, column, a.size(1), b.size(1));
            assertEquals(expected, actual[row * b.size(1) + column], precision * Math.abs(expected) + 2e-6,
                    "row=" + row + ", column=" + column + ", dtype=" + a.dtype() + "/" + b.dtype());
        }
    }

    private static double dot(float[] a, float[] b, int row, int column, int k, int n) {
        double sum = 0;
        for (int i = 0; i < k; i++) { sum += (double) a[row * k + i] * b[i * n + column]; }
        return sum;
    }

    private static void checkLeftGradient(Tensor a, Tensor b) {
        for (int k = 0; k < a.size(1); k++) {
            double expected = 0;
            for (int n = 0; n < b.size(1); n++) { expected += b.get(k, n); }
            for (int row = 0; row < a.size(0); row++) {
                assertEquals(expected, a.grad().get(row, k), 2e-5 * Math.abs(expected) + 2e-6);
            }
        }
    }

    private static void checkRightGradient(Tensor a, Tensor b) {
        for (int k = 0; k < b.size(0); k++) {
            double expected = 0;
            for (int row = 0; row < a.size(0); row++) { expected += a.get(row, k); }
            for (int n = 0; n < b.size(1); n++) { assertEquals(expected, b.grad().get(k, n), 2e-6); }
        }
    }

    private static Tensor values(DType dtype, int rows, int columns) {
        var random = new Random(37L * rows + columns);
        float[] values = new float[rows * columns];
        for (int i = 0; i < values.length; i++) { values[i] = random.nextFloat() * .2f - .1f; }
        return Tensor.from(values, dtype, rows, columns);
    }
}
