package test;

import org.DeepJ.transformer.LayerNorm;
import org.DeepJ.transformer.Tensor;

public class FiniteDiffLayerNormTest {

    public static void main(String[] args) {

        int rows = 2, cols = 3;
        LayerNorm ln = new LayerNorm(cols);
        ln.setLearningRate(0.0);                  // we will NOT update weights here

        // small random input so numbers aren’t too large
        Tensor x = new Tensor(new double[][]{
                { 0.3, -1.2, 2.1 },
                { 1.7,  0.5, -0.6 }
        });

        /* ------------ forward + analytical gradients ------------ */
        Tensor y     = ln.forward(x);

        // pretend loss = sum(y)  → upstream grad is all 1 s
        Tensor upstream = Tensor.ones(rows, cols);

        Tensor dX      = ln.backward(upstream);   // analytic ∂L/∂x
        Tensor dGamma  = upstream.multiply(ln.getNormalized()).sumAlongCols(); // analytic ∂L/∂γ
        Tensor dBeta   = upstream.sumAlongCols(); // analytic ∂L/∂β

        /* ------------ numerical gradients ------------ */
        double eps = 1e-4;

        // ∂L/∂x (finite diff)
        Tensor num_dX = new Tensor(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double old = x.data[r][c];

                x.data[r][c] = old + eps;
                double plus = ln.forward(x).sum();   // Σy
                x.data[r][c] = old - eps;
                double minus = ln.forward(x).sum();
                x.data[r][c] = old;                  // restore

                num_dX.data[r][c] = (plus - minus) / (2*eps);
            }
        }

        // ∂L/∂γ (finite diff)
        Tensor num_dGamma = new Tensor(1, cols);
        for (int c = 0; c < cols; c++) {
            double old = ln.getGamma().data[0][c];

            ln.getGamma().data[0][c] = old + eps;
            double plus = ln.forward(x).sum();
            ln.getGamma().data[0][c] = old - eps;
            double minus = ln.forward(x).sum();
            ln.getGamma().data[0][c] = old;

            num_dGamma.data[0][c] = (plus - minus) / (2*eps);
        }

        // ∂L/∂β (finite diff)
        Tensor num_dBeta = new Tensor(1, cols);
        for (int c = 0; c < cols; c++) {
            double old = ln.getBeta().data[0][c];

            ln.getBeta().data[0][c] = old + eps;
            double plus = ln.forward(x).sum();
            ln.getBeta().data[0][c] = old - eps;
            double minus = ln.forward(x).sum();
            ln.getBeta().data[0][c] = old;

            num_dBeta.data[0][c] = (plus - minus) / (2*eps);
        }

        /* ------------ compare ------------ */
        assert closeEnough(dX, num_dX)        : "∂L/∂x mismatch";
        assert closeEnough(dGamma, num_dGamma): "∂L/∂γ mismatch";
        assert closeEnough(dBeta,  num_dBeta) : "∂L/∂β mismatch";

        System.out.println("✅ finite-difference gradient check passed");
    }

    private static boolean closeEnough(Tensor a, Tensor b) {
        double tol = 1e-3;                    // ~ 0.1 % relative error
        for (int r = 0; r < a.rows; r++)
            for (int c = 0; c < a.cols; c++) {
                double denom = Math.max(1.0, Math.abs(a.data[r][c]));
                if (Math.abs(a.data[r][c] - b.data[r][c]) / denom > tol)
                    return false;
            }
        return true;
    }
}
