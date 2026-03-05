package io.github.kirstenali.deepj;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;

public final class TestSupport {

    private TestSupport() {}

    public static void assertTensorShape(Tensor t, int rows, int cols) {
        Assertions.assertNotNull(t, "tensor must not be null");
        Assertions.assertEquals(rows, t.rows, "rows");
        Assertions.assertEquals(cols, t.cols, "cols");
    }

    public static void assertTensorAllClose(Tensor a, Tensor b, double tol) {
        Assertions.assertEquals(a.rows, b.rows, "rows");
        Assertions.assertEquals(a.cols, b.cols, "cols");
        for (int r = 0; r < a.rows; r++) {
            for (int c = 0; c < a.cols; c++) {
                double av = a.data[r][c];
                double bv = b.data[r][c];
                if (Double.isNaN(av) || Double.isNaN(bv)) {
                    Assertions.fail("NaN encountered at [" + r + "," + c + "]");
                }
                Assertions.assertTrue(Math.abs(av - bv) <= tol,
                        "Mismatch at [" + r + "," + c + "]: " + av + " vs " + bv);
            }
        }
    }

    public static Tensor tensor(double[][] data) {
        return new Tensor(data);
    }

    public static double rowMean(Tensor t, int r) {
        double s = 0.0;
        for (int c = 0; c < t.cols; c++) s += t.data[r][c];
        return s / t.cols;
    }

    public static double rowVar(Tensor t, int r) {
        double mean = rowMean(t, r);
        double s = 0.0;
        for (int c = 0; c < t.cols; c++) {
            double d = t.data[r][c] - mean;
            s += d * d;
        }
        return s / t.cols;
    }

    public static double tensorSumAbs(Tensor t) {
        double s = 0.0;
        for (int r = 0; r < t.rows; r++) {
            for (int c = 0; c < t.cols; c++) s += Math.abs(t.data[r][c]);
        }
        return s;
    }

    public static Tensor copy(Tensor t) {
        return new Tensor(t.data);
    }
}
