package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link SiLU}.
 *
 * <p>Covers:
 * <ul>
 *   <li>Known forward values (SiLU(x) = x · σ(x))</li>
 *   <li>Shape preservation in forward and backward</li>
 *   <li>Backward: known gradient values</li>
 *   <li>Numerical gradient check (finite differences vs analytical)</li>
 *   <li>Guard: backward before forward throws</li>
 *   <li>Guard: shape mismatch in backward throws</li>
 * </ul>
 */
class SiLUTest {

    private SiLU silu;

    @BeforeEach
    void setUp() {
        silu = new SiLU();
    }

    // ── forward ──────────────────────────────────────────────────────────────

    @Test
    void forward_zero_gives_zero() {
        Tensor x = new Tensor(new double[][]{{0.0, 0.0}});
        Tensor y = silu.forward(x);
        TestSupport.assertTensorAllClose(Tensor.zeros(1, 2), y, 1e-12);
    }

    @Test
    void forward_known_positive_value() {
        // SiLU(1) = 1 · σ(1) = 0.7310585786...
        double expected = 1.0 / (1.0 + Math.exp(-1.0));
        Tensor x = new Tensor(new double[][]{{1.0}});
        Tensor y = silu.forward(x);
        assertEquals(expected, y.data[0], 1e-9);
    }

    @Test
    void forward_known_negative_value() {
        // SiLU(-1) = -1 · σ(-1) = -0.2689414213...
        double sig = 1.0 / (1.0 + Math.exp(1.0));
        double expected = -1.0 * sig;
        Tensor x = new Tensor(new double[][]{{-1.0}});
        Tensor y = silu.forward(x);
        assertEquals(expected, y.data[0], 1e-9);
    }

    @Test
    void forward_preserves_shape() {
        Tensor x = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor y = silu.forward(x);
        TestSupport.assertTensorShape(y, 2, 3);
    }

    @Test
    void forward_positive_inputs_are_positive() {
        Tensor x = new Tensor(new double[][]{{0.5, 1.0, 2.0, 5.0}});
        Tensor y = silu.forward(x);
        for (int c = 0; c < y.cols; c++) {
            assertTrue(y.data[c] > 0, "SiLU of positive input should be positive");
        }
    }

    // ── backward ─────────────────────────────────────────────────────────────

    @Test
    void backward_at_zero_is_half() {
        // SiLU'(0) = σ(0) + 0 · σ(0) · (1 − σ(0)) = 0.5
        Tensor x = new Tensor(new double[][]{{0.0}});
        silu.forward(x);
        Tensor grad = silu.backward(Tensor.ones(1, 1));
        assertEquals(0.5, grad.data[0], 1e-9);
    }

    @Test
    void backward_preserves_shape() {
        Tensor x = new Tensor(new double[][]{{1.0, -1.0, 2.0}, {0.0, 3.0, -2.0}});
        silu.forward(x);
        Tensor grad = silu.backward(Tensor.ones(2, 3));
        TestSupport.assertTensorShape(grad, 2, 3);
    }

    @Test
    void backward_numerical_gradient_check() {
        double[] vals = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0};
        double eps = 1e-5;

        for (double v : vals) {
            SiLU s1 = new SiLU();
            SiLU s2 = new SiLU();
            double plus  = s1.forward(new Tensor(new double[][]{{v + eps}})).data[0];
            double minus = s2.forward(new Tensor(new double[][]{{v - eps}})).data[0];
            double numerical = (plus - minus) / (2 * eps);

            SiLU analytic = new SiLU();
            analytic.forward(new Tensor(new double[][]{{v}}));
            double analytical = analytic.backward(Tensor.ones(1, 1)).data[0];

            assertEquals(numerical, analytical, 1e-6,
                    "Gradient mismatch at x=" + v);
        }
    }

    @Test
    void backward_gradOut_scales_result() {
        Tensor x = new Tensor(new double[][]{{1.0}});
        silu.forward(x);
        Tensor g1 = silu.backward(Tensor.ones(1, 1));

        silu = new SiLU();
        silu.forward(x);
        Tensor g3 = silu.backward(new Tensor(new double[][]{{3.0}}));

        assertEquals(g1.data[0] * 3.0, g3.data[0], 1e-12);
    }

    // ── guards ───────────────────────────────────────────────────────────────

    @Test
    void backward_before_forward_throws() {
        assertThrows(IllegalStateException.class,
                () -> silu.backward(Tensor.ones(1, 1)));
    }

    @Test
    void backward_shape_mismatch_throws() {
        silu.forward(new Tensor(new double[][]{{1.0, 2.0}}));
        assertThrows(IllegalArgumentException.class,
                () -> silu.backward(new Tensor(new double[][]{{1.0}})));
    }
}

