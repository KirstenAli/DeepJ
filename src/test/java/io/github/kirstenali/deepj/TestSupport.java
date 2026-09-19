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
                double av = a.data[r * a.cols + c];
                double bv = b.data[r * b.cols + c];
                if (Double.isNaN(av) || Double.isNaN(bv)) {
                    Assertions.fail("NaN encountered at [" + r + "," + c + "]");
                }
                Assertions.assertTrue(Math.abs(av - bv) <= tol,
                        "Mismatch at [" + r + "," + c + "]: " + av + " vs " + bv);
            }
        }
    }
}
