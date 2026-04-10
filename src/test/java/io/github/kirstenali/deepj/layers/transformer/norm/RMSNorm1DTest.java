package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;

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

    private static final float EPS = 1e-6f;

    private RMSNorm1D norm;

    @BeforeEach
    void setUp() {
        norm = new RMSNorm1D(4);
    }

    // ── forward ──────────────────────────────────────────────────────────────

    @Test
    void forward_preserves_shape() {
        Tensor x = Tensor.from2D(new float[][]{{1, 2, 3, 4}, {-1, 0, 1, 2}});
        Tensor y = norm.forward(x);
        TestSupport.assertTensorShape(y, 2, 4);
    }

    @Test
    void forward_no_nan_or_inf() {
        Tensor x = Tensor.from2D(new float[][]{{0.1f, -0.3f, 0.5f, -0.2f}, {3.0f, 1.0f, 2.0f, 4.0f}});
        Tensor y = norm.forward(x);
        for (int r = 0; r < y.rows; r++) {
            for (int c = 0; c < y.cols; c++) {
                assertTrue(Double.isFinite(y.data[r * y.cols + c]), "output must be finite at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forward_with_identity_gamma_normalises_to_unit_rms() {
        // With gamma = ones, each output row should have RMS ≈ 1.f
        Tensor x = Tensor.from2D(new float[][]{{3.0f, 0.0f, 4.0f, 0.0f}});  // RMS = sqrt((9+16)/4) = 2.5f
        Tensor y = norm.forward(x);

        double sumSq = 0;
        for (int c = 0; c < y.cols; c++) sumSq += y.data[c] * y.data[c];
        double rms = Math.sqrt(sumSq / y.cols);
        assertEquals(1.0f, rms, 1e-5f);
    }

    @Test
    void forward_gamma_scaling_doubles_output() {
        // Set gamma = 2 × ones; output should be exactly 2× the unit-normalised value.
        norm = new RMSNorm1D(4);
        Tensor x = Tensor.from2D(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}});

        Tensor y1 = norm.forward(x);  // gamma = ones

        // Scale gamma to 2
        norm = new RMSNorm1D(4);
        for (Parameter p : norm.parameters()) {
            p.value = Tensor.from2D(new float[][]{{2.0f, 2.0f, 2.0f, 2.0f}});
        }
        Tensor y2 = norm.forward(x);

        for (int c = 0; c < y1.cols; c++) {
            assertEquals(y1.data[c] * 2.0f, y2.data[c], 1e-9f);
        }
    }

    @Test
    void forward_constant_input_normalises_to_plus_or_minus_one() {
        Tensor x = Tensor.from2D(new float[][]{{5.0f, 5.0f, 5.0f, 5.0f}});
        Tensor y = norm.forward(x);
        for (int c = 0; c < y.cols; c++) {
            assertEquals(1.0f, y.data[c], 1e-5f);
        }
    }

    @Test
    void forward_multi_row_independent_normalisation() {
        // Scaling the input by a constant should produce nearly the same normalised output.
        // Exact equality holds only when eps=0; with eps=1e-6f the difference is ~O(eps/rms²),
        // so we allow 1e-6f tolerance.
        Tensor x  = Tensor.from2D(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}});
        Tensor x2 = Tensor.from2D(new float[][]{{2.0f, 4.0f, 6.0f, 8.0f}});
        Tensor y1 = norm.forward(x);
        norm = new RMSNorm1D(4);   // fresh instance so caches are clean
        Tensor y2 = norm.forward(x2);
        TestSupport.assertTensorAllClose(y1, y2, 1e-6f);
    }

    // ── backward ─────────────────────────────────────────────────────────────

    @Test
    void backward_returns_correct_shape() {
        Tensor x = Tensor.from2D(new float[][]{{1.0f, -1.0f, 2.0f, 0.5f}, {0.0f, 3.0f, -2.0f, 1.0f}});
        norm.forward(x);
        Tensor grad = norm.backward(Tensor.ones(2, 4));
        TestSupport.assertTensorShape(grad, 2, 4);
    }

    @Test
    void backward_accumulates_gamma_gradient() {
        Tensor x = Tensor.from2D(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}});
        norm.forward(x);
        norm.backward(Tensor.ones(1, 4));

        Parameter gamma = norm.parameters().get(0);
        double totalGrad = 0;
        for (int c = 0; c < 4; c++) totalGrad += Math.abs(gamma.grad.data[c]);
        assertTrue(totalGrad > 0, "gamma gradient should be non-zero after backward");
    }

    @Test
    void backward_gamma_gradient_is_sum_of_gradOut_times_xHat() {
        Tensor x = Tensor.from2D(new float[][]{{2.0f, 0.0f, 2.0f, 0.0f}});
        norm.forward(x);
        Tensor gradOut = Tensor.ones(1, 4);
        norm.backward(gradOut);

        double rms = Math.sqrt(2.0f + EPS);
        Parameter gamma = norm.parameters().get(0);
        assertEquals(2.0f / rms, gamma.grad.data[0], 1e-5f);
        assertEquals(0.0f,       gamma.grad.data[1], 1e-5f);
    }

    @Test
    void backward_numerical_gradient_check() {
        float finiteDiffEps = 1e-3f;
        float tolerance     = 2e-3f;

        Tensor x = Tensor.from2D(new float[][]{{0.5f, -1.0f, 2.0f, 0.3f}});

        norm.forward(x);
        Tensor analyticGrad = norm.backward(Tensor.ones(1, 4));

        for (int c = 0; c < 4; c++) {
            float orig = x.get(0, c);

            x.set(0, c, orig + finiteDiffEps);
            float fPlus = sumAll(new RMSNorm1D(4).forward(x));

            x.set(0, c, orig - finiteDiffEps);
            float fMinus = sumAll(new RMSNorm1D(4).forward(x));

            x.set(0, c, orig);

            float numerical = (fPlus - fMinus) / (2 * finiteDiffEps);
            assertEquals(numerical, analyticGrad.get(0, c), tolerance,
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
            assertEquals(1.0f, gamma.value.data[c], 1e-12f);
        }
    }

    // ── guards ───────────────────────────────────────────────────────────────

    @Test
    void forward_wrong_cols_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> norm.forward(Tensor.from2D(new float[][]{{1.0f, 2.0f}})));
    }

    @Test
    void constructor_zero_dim_throws() {
        assertThrows(IllegalArgumentException.class, () -> new RMSNorm1D(0));
    }

    // ── helper ───────────────────────────────────────────────────────────────

    private static float sumAll(Tensor t) {
        float s = 0.0f;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                s += t.data[r * t.cols + c];
        return s;
    }
}
