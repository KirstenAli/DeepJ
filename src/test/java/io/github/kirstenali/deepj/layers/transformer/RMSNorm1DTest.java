package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RMSNorm1D}.
 *
 * <p>Covers:
 * <ul>
 *   <li>Forward: output has unit RMS (before gamma scaling)</li>
 *   <li>Forward: gamma correctly scales the normalised output</li>
 *   <li>Forward: shape preserved, no NaN</li>
 *   <li>Backward: correct shape returned</li>
 *   <li>Backward: gamma gradient accumulated</li>
 *   <li>Backward: numerical gradient check (finite differences)</li>
 *   <li>No beta parameter — only gamma exposed</li>
 *   <li>Guard: wrong input width throws</li>
 * </ul>
 */
class RMSNorm1DTest {

    private static final double EPS = 1e-6;

    private RMSNorm1D norm;

    @BeforeEach
    void setUp() {
        norm = new RMSNorm1D(4);
    }

    // ── forward ──────────────────────────────────────────────────────────────

    @Test
    void forward_preserves_shape() {
        Tensor x = new Tensor(new double[][]{{1, 2, 3, 4}, {-1, 0, 1, 2}});
        Tensor y = norm.forward(x);
        TestSupport.assertTensorShape(y, 2, 4);
    }

    @Test
    void forward_no_nan_or_inf() {
        Tensor x = new Tensor(new double[][]{{0.1, -0.3, 0.5, -0.2}, {3.0, 1.0, 2.0, 4.0}});
        Tensor y = norm.forward(x);
        for (int r = 0; r < y.rows; r++) {
            for (int c = 0; c < y.cols; c++) {
                assertTrue(Double.isFinite(y.data[r][c]), "output must be finite at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forward_with_identity_gamma_normalises_to_unit_rms() {
        // With gamma = ones, each output row should have RMS ≈ 1.
        Tensor x = new Tensor(new double[][]{{3.0, 0.0, 4.0, 0.0}});  // RMS = sqrt((9+16)/4) = 2.5
        Tensor y = norm.forward(x);

        double sumSq = 0;
        for (int c = 0; c < y.cols; c++) sumSq += y.data[0][c] * y.data[0][c];
        double rms = Math.sqrt(sumSq / y.cols);
        assertEquals(1.0, rms, 1e-5);
    }

    @Test
    void forward_gamma_scaling_doubles_output() {
        // Set gamma = 2 × ones; output should be exactly 2× the unit-normalised value.
        norm = new RMSNorm1D(4);
        Tensor x = new Tensor(new double[][]{{1.0, 2.0, 3.0, 4.0}});

        Tensor y1 = norm.forward(x);  // gamma = ones

        // Scale gamma to 2
        norm = new RMSNorm1D(4);
        for (Parameter p : norm.parameters()) {
            p.value = new Tensor(new double[][]{{2.0, 2.0, 2.0, 2.0}});
        }
        Tensor y2 = norm.forward(x);

        for (int c = 0; c < y1.cols; c++) {
            assertEquals(y1.data[0][c] * 2.0, y2.data[0][c], 1e-9);
        }
    }

    @Test
    void forward_constant_input_normalises_to_plus_or_minus_one() {
        // All same non-zero value: xHat_i = 1 for all i; out_i = gamma_i = 1.
        Tensor x = new Tensor(new double[][]{{5.0, 5.0, 5.0, 5.0}});
        Tensor y = norm.forward(x);
        for (int c = 0; c < y.cols; c++) {
            assertEquals(1.0, y.data[0][c], 1e-5);
        }
    }

    @Test
    void forward_multi_row_independent_normalisation() {
        // Scaling the input by a constant should produce nearly the same normalised output.
        // Exact equality holds only when eps=0; with eps=1e-6 the difference is ~O(eps/rms²),
        // so we allow 1e-6 tolerance.
        Tensor x  = new Tensor(new double[][]{{1.0, 2.0, 3.0, 4.0}});
        Tensor x2 = new Tensor(new double[][]{{2.0, 4.0, 6.0, 8.0}});
        Tensor y1 = norm.forward(x);
        norm = new RMSNorm1D(4);   // fresh instance so caches are clean
        Tensor y2 = norm.forward(x2);
        TestSupport.assertTensorAllClose(y1, y2, 1e-6);
    }

    // ── backward ─────────────────────────────────────────────────────────────

    @Test
    void backward_returns_correct_shape() {
        Tensor x = new Tensor(new double[][]{{1.0, -1.0, 2.0, 0.5}, {0.0, 3.0, -2.0, 1.0}});
        norm.forward(x);
        Tensor grad = norm.backward(Tensor.ones(2, 4));
        TestSupport.assertTensorShape(grad, 2, 4);
    }

    @Test
    void backward_accumulates_gamma_gradient() {
        Tensor x = new Tensor(new double[][]{{1.0, 2.0, 3.0, 4.0}});
        norm.forward(x);
        norm.backward(Tensor.ones(1, 4));

        Parameter gamma = norm.parameters().get(0);
        double totalGrad = 0;
        for (int c = 0; c < 4; c++) totalGrad += Math.abs(gamma.grad.data[0][c]);
        assertTrue(totalGrad > 0, "gamma gradient should be non-zero after backward");
    }

    @Test
    void backward_gamma_gradient_is_sum_of_gradOut_times_xHat() {
        // grad_gamma[c] = sum_rows(gradOut[r,c] * xHat[r,c])
        Tensor x = new Tensor(new double[][]{{2.0, 0.0, 2.0, 0.0}});
        norm.forward(x);
        Tensor gradOut = Tensor.ones(1, 4);
        norm.backward(gradOut);

        // With ones gradOut, grad_gamma[c] = xHat[c]
        // For input [2, 0, 2, 0]: rms = sqrt((4+0+4+0)/4) = sqrt(2)
        // xHat = [2/sqrt(2), 0, 2/sqrt(2), 0] = [sqrt(2), 0, sqrt(2), 0]
        double rms = Math.sqrt(2.0 + EPS);
        Parameter gamma = norm.parameters().get(0);
        assertEquals(2.0 / rms, gamma.grad.data[0][0], 1e-5);
        assertEquals(0.0,       gamma.grad.data[0][1], 1e-5);
    }

    @Test
    void backward_numerical_gradient_check() {
        double finiteDiffEps = 1e-5;
        double tolerance     = 1e-4;

        Tensor x = new Tensor(new double[][]{{0.5, -1.0, 2.0, 0.3}});

        // Analytical gradient (backward with gradOut = ones)
        norm.forward(x);
        Tensor analyticGrad = norm.backward(Tensor.ones(1, 4));

        // Numerical gradient via finite differences on sum(forward(x))
        for (int c = 0; c < 4; c++) {
            double orig = x.data[0][c];

            x.data[0][c] = orig + finiteDiffEps;
            double fPlus = sumAll(new RMSNorm1D(4).forward(x));

            x.data[0][c] = orig - finiteDiffEps;
            double fMinus = sumAll(new RMSNorm1D(4).forward(x));

            x.data[0][c] = orig;  // restore

            double numerical = (fPlus - fMinus) / (2 * finiteDiffEps);
            assertEquals(numerical, analyticGrad.data[0][c], tolerance,
                    "Gradient mismatch at col=" + c);
        }
    }

    // ── parameters ───────────────────────────────────────────────────────────

    @Test
    void has_exactly_one_parameter_gamma_no_beta() {
        assertEquals(1, norm.parameters().size(), "RMSNorm1D must expose only gamma (no beta)");
    }

    @Test
    void gamma_initialised_to_ones() {
        Parameter gamma = norm.parameters().get(0);
        for (int c = 0; c < 4; c++) {
            assertEquals(1.0, gamma.value.data[0][c], 1e-12);
        }
    }

    // ── guards ───────────────────────────────────────────────────────────────

    @Test
    void forward_wrong_cols_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> norm.forward(new Tensor(new double[][]{{1.0, 2.0}})));
    }

    @Test
    void constructor_zero_dim_throws() {
        assertThrows(IllegalArgumentException.class, () -> new RMSNorm1D(0));
    }

    // ── helper ───────────────────────────────────────────────────────────────

    private static double sumAll(Tensor t) {
        double s = 0;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                s += t.data[r][c];
        return s;
    }
}
