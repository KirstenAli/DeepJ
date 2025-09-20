package org.DeepJ;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.layers.LayerNorm;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LayerNormTest {
    @Test
    public void rowMeansAreZero_whenGamma1Beta0() {
        int rows = 4, dim = 6;
        LayerNorm ln = new LayerNorm(dim);
        ln.setGamma(Tensor.ones(1, dim));
        ln.setBeta (Tensor.zeros(1, dim));

        Tensor x = Tensor.random(rows, dim, new java.util.Random(1234));
        Tensor y = ln.forward(x);
        Tensor rowMeans = y.meanAlongRows(); // rows × 1

        for (int i = 0; i < rows; i++) {
            double mean = rowMeans.data[i][0];
            assertClose(0.0, mean, 1e-6, 0.0, "Row " + i + " mean not ~0");
        }
    }

    @Test
    public void rowVarsAreOne_whenGamma1Beta0() {
        int rows = 4, dim = 6;
        LayerNorm ln = new LayerNorm(dim);
        ln.setGamma(Tensor.ones(1, dim));
        ln.setBeta (Tensor.zeros(1, dim));

        Tensor x = Tensor.random(rows, dim, new java.util.Random(42));
        Tensor y = ln.forward(x);
        Tensor rowVars = y.varianceAlongRows(); // rows × 1

        for (int i = 0; i < rows; i++) {
            double var = rowVars.data[i][0];
            assertClose(1.0, var, 1e-4, 2e-3, "Row " + i + " variance not ~1");
        }
    }

    @Test
    public void affineParamsShiftAndScaleMeansAndVars() {
        int rows = 4, dim = 6;
        LayerNorm ln = new LayerNorm(dim);

        // gamma = 2, beta = 3
        ln.setGamma(fill(1, dim, 2.0));
        ln.setBeta (fill(1, dim, 3.0));

        Tensor x = Tensor.random(rows, dim, new java.util.Random(7));
        Tensor y = ln.forward(x);

        Tensor rowMeans = y.meanAlongRows();
        Tensor rowVars  = y.varianceAlongRows();

        for (int i = 0; i < rows; i++) {
            double m = rowMeans.data[i][0];
            double v = rowVars.data[i][0];

            assertClose(3.0, m, 1e-3, 0.0, "Row " + i + " mean not ~3");
            assertClose(4.0, v, 1e-6, 5e-3, "Row " + i + " variance not ~4");
        }
    }

    @Test
    public void invariantToPerRowAdditiveShift_whenGamma1Beta0() {
        int rows = 4, dim = 6;
        LayerNorm ln = new LayerNorm(dim);
        ln.setGamma(Tensor.ones(1, dim));
        ln.setBeta (Tensor.zeros(1, dim));

        Tensor x = Tensor.random(rows, dim, new java.util.Random(999));
        Tensor y = ln.forward(x);

        // Per-row shifts: rows×1 column vector broadcast across columns
        Tensor shifts = new Tensor(new double[][]{
                {+3.0},
                {-1.2},
                {+0.5},
                {-7.7}
        });
        Tensor xShifted = x.addBroadcastCols(shifts);
        Tensor yShifted = ln.forward(xShifted);

        assertTrue(allClose(y, yShifted, 1e-6, 1e-6),
                "Output should be invariant to per-row additive shifts");
    }

    @Test
    public void gradcheck_LayerNorm_sumLoss() {
        int rows = 2, dim = 3;
        double eps = 1e-5, tol = 1e-3;

        LayerNorm ln = new LayerNorm(dim);
        ln.setGamma(new Tensor(new double[][]{{1.1, 0.9, 1.3}}));
        ln.setBeta (new Tensor(new double[][]{{-0.2, 0.1, 0.3}}));

        Tensor x = new Tensor(new double[][]{
                {0.3, -0.7, 1.1},
                {-0.4, 0.2, 0.5}
        });

        // forward + backward with dL/dY = 1
        Tensor y = ln.forward(x);
        Tensor dLdY = Tensor.ones(rows, dim);
        Tensor dLdX = ln.backward(dLdY, 0.0);

        // finite-diff w.r.t. x
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < dim; j++) {
                double orig = x.data[i][j];

                x.data[i][j] = orig + eps;
                double Lp = ln.forward(x).sum();

                x.data[i][j] = orig - eps;
                double Lm = ln.forward(x).sum();

                x.data[i][j] = orig; // restore

                double num = (Lp - Lm) / (2*eps);
                double ana = dLdX.data[i][j];
                assertEquals(num, ana, tol, "dx mismatch at ("+i+","+j+")");
            }
        }
    }

    private static void assertClose(double expected, double actual, double atol, double rtol, String msg) {
        double diff = Math.abs(expected - actual);
        double tol  = atol + rtol * Math.max(Math.abs(expected), Math.abs(actual));
        if (diff > tol) {
            fail(String.format("%s | expected=%.9f actual=%.9f diff=%.3e tol=%.3e",
                    msg, expected, actual, diff, tol));
        }
    }

    private static boolean allClose(Tensor a, Tensor b, double atol, double rtol) {
        if (a.rows != b.rows || a.cols != b.cols) return false;
        final boolean[] ok = {true};
        a.iterate((r, c) -> {
            double x = a.data[r][c], y = b.data[r][c];
            double diff = Math.abs(x - y);
            double tol  = atol + rtol * Math.max(Math.abs(x), Math.abs(y));
            if (diff > tol) ok[0] = false;
        });
        return ok[0];
    }

    private static Tensor fill(int rows, int cols, double v) {
        Tensor t = new Tensor(rows, cols);
        t.iterate((r, c) -> t.data[r][c] = v);
        return t;
    }
}
