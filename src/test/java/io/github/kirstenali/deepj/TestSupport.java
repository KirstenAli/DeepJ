package io.github.kirstenali.deepj;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;

import java.util.function.DoubleSupplier;

public final class TestSupport {

    private TestSupport() {}

    public static void assertTensorShape(Tensor t, int rows, int cols) {
        Assertions.assertNotNull(t, "tensor must not be null");
        Assertions.assertEquals(rows, t.matrixRows(), "rows");
        Assertions.assertEquals(cols, t.matrixCols(), "cols");
    }

    public static void assertTensorAllClose(Tensor a, Tensor b, double tol) {
        Assertions.assertEquals(a.matrixRows(), b.matrixRows(), "rows");
        Assertions.assertEquals(a.matrixCols(), b.matrixCols(), "cols");
        for (int r = 0; r < a.matrixRows(); r++) {
            for (int c = 0; c < a.matrixCols(); c++) {
                double av = a.rawData()[r * a.matrixCols() + c];
                double bv = b.rawData()[r * b.matrixCols() + c];
                if (Double.isNaN(av) || Double.isNaN(bv)) {
                    Assertions.fail("NaN encountered at [" + r + "," + c + "]");
                }
                Assertions.assertTrue(Math.abs(av - bv) <= tol,
                        "Mismatch at [" + r + "," + c + "]: " + av + " vs " + bv);
            }
        }
    }

    public static void assertLossDecreases(DoubleSupplier trainingStep, int attempts, String message) {
        double previous = trainingStep.getAsDouble();
        for (int attempt = 0; attempt < attempts; attempt++) {
            double current = trainingStep.getAsDouble();
            if (current < previous) return;
            previous = current;
        }
        Assertions.fail(message);
    }
}
