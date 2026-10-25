package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.testing.AutogradTestSupport;
import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SiLUTest {

    private SiLU silu;

    @BeforeEach
    void setUp() {
        silu = new SiLU();
    }

    @Test
    void forward_zero_gives_zero() {
        Tensor x = Tensor.from2D(new float[][]{{0.0f, 0.0f}});
        Tensor y = silu.forward(x);
        TestSupport.assertTensorAllClose(Tensor.zeros(1, 2), y, 1e-6f);
    }

    @Test
    void forward_known_positive_value() {

        float expected = (float) (1.0f / (1.0f + Math.exp(-1.0f)));
        Tensor x = Tensor.from2D(new float[][]{{1.0f}});
        Tensor y = silu.forward(x);
        assertEquals(expected, y.rawData()[0], 1e-6f);
    }

    @Test
    void forward_known_negative_value() {

        float sig = (float) (1.0f / (1.0f + Math.exp(1.0f)));
        float expected = -1.0f * sig;
        Tensor x = Tensor.from2D(new float[][]{{-1.0f}});
        Tensor y = silu.forward(x);
        assertEquals(expected, y.rawData()[0], 1e-6f);
    }

    @Test
    void forward_preserves_shape() {
        Tensor x = Tensor.from2D(new float[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor y = silu.forward(x);
        TestSupport.assertTensorShape(y, 2, 3);
    }

    @Test
    void forward_positive_inputs_are_positive() {
        Tensor x = Tensor.from2D(new float[][]{{0.5f, 1.0f, 2.0f, 5.0f}});
        Tensor y = silu.forward(x);
        for (int c = 0; c < y.cols; c++) {
            assertTrue(y.rawData()[c] > 0, "SiLU of positive input should be positive");
        }
    }

    @Test
    void backward_at_zero_is_half() {

        Tensor x = Tensor.from2D(new float[][]{{0.0f}});
        silu.forward(x);
        Tensor grad = AutogradTestSupport.backward(silu::forward, x, Tensor.ones(1, 1));
        assertEquals(0.5f, grad.rawData()[0], 1e-6f);
    }

    @Test
    void backward_preserves_shape() {
        Tensor x = Tensor.from2D(new float[][]{{1.0f, -1.0f, 2.0f}, {0.0f, 3.0f, -2.0f}});
        silu.forward(x);
        Tensor grad = AutogradTestSupport.backward(silu::forward, x, Tensor.ones(2, 3));
        TestSupport.assertTensorShape(grad, 2, 3);
    }

    @Test
    void backward_numerical_gradient_check() {
        float[] vals = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
        float eps = 1e-3f;

        for (float v : vals) {
            SiLU s1 = new SiLU();
            SiLU s2 = new SiLU();
            float plus  = s1.forward(Tensor.from2D(new float[][]{{v + eps}})).rawData()[0];
            float minus = s2.forward(Tensor.from2D(new float[][]{{v - eps}})).rawData()[0];
            float numerical = (plus - minus) / (2 * eps);

            SiLU analytic = new SiLU();
            analytic.forward(Tensor.from2D(new float[][]{{v}}));
            Tensor input = Tensor.from2D(new float[][]{{v}});
            float analytical = AutogradTestSupport.backward(analytic::forward, input, Tensor.ones(1, 1)).rawData()[0];

            assertEquals(numerical, analytical, 2e-3f,
                    "Gradient mismatch at x=" + v);
        }
    }

    @Test
    void backward_gradOut_scales_result() {
        Tensor x = Tensor.from2D(new float[][]{{1.0f}});
        silu.forward(x);
        Tensor g1 = AutogradTestSupport.backward(silu::forward, x, Tensor.ones(1, 1));

        silu = new SiLU();
        silu.forward(x);
        Tensor g3 = AutogradTestSupport.backward(silu::forward, x, Tensor.from2D(new float[][]{{3.0f}}));

        assertEquals(g1.rawData()[0] * 3.0f, g3.rawData()[0], 1e-6f);
    }

    @Test
    void backward_without_gradient_tracking_throws() {
        assertThrows(IllegalStateException.class,
                () -> silu.forward(Tensor.ones(1, 1)).backward());
    }

    @Test
    void backward_shape_mismatch_throws() {
        silu.forward(Tensor.from2D(new float[][]{{1.0f, 2.0f}}));
        Tensor input = Tensor.from2D(new float[][]{{1.0f, 2.0f}});
        assertThrows(IllegalArgumentException.class,
                () -> AutogradTestSupport.backward(silu::forward, input, Tensor.ones(1, 1)));
    }
}
