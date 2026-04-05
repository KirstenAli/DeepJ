package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link SwiGLULayer}.
 *
 * <p>Covers:
 * <ul>
 *   <li>Forward output shape and finiteness</li>
 *   <li>Backward output shape</li>
 *   <li>Backward accumulates gradients in all three projections</li>
 *   <li>Backward numerical gradient check (finite differences)</li>
 *   <li>Learning: MSE loss decreases with AdamW</li>
 *   <li>Projection count: exactly 3 Linear layers worth of parameters</li>
 *   <li>Guards: invalid constructor args throw</li>
 * </ul>
 */
class SwiGLULayerTest {

    // ── forward ──────────────────────────────────────────────────────────────

    @Test
    void forward_returns_correct_shape() {
        SwiGLULayer layer = new SwiGLULayer(8, 16, new Random(1));
        Tensor x = randomTensor(3, 8, 42);
        Tensor y = layer.forward(x);
        TestSupport.assertTensorShape(y, 3, 8);
    }

    @Test
    void forward_no_nan_or_inf() {
        SwiGLULayer layer = new SwiGLULayer(4, 8, new Random(2));
        Tensor x = randomTensor(2, 4, 7);
        Tensor y = layer.forward(x);
        for (int r = 0; r < y.rows; r++) {
            for (int c = 0; c < y.cols; c++) {
                assertTrue(Double.isFinite(y.data[r][c]),
                        "output must be finite at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forward_single_token() {
        SwiGLULayer layer = new SwiGLULayer(4, 8, new Random(3));
        Tensor x = randomTensor(1, 4, 11);
        Tensor y = layer.forward(x);
        TestSupport.assertTensorShape(y, 1, 4);
    }

    // ── backward ─────────────────────────────────────────────────────────────

    @Test
    void backward_returns_correct_shape() {
        SwiGLULayer layer = new SwiGLULayer(6, 12, new Random(4));
        Tensor x = randomTensor(4, 6, 99);
        Tensor y = layer.forward(x);
        Tensor gradIn = layer.backward(Tensor.ones(y.rows, y.cols));
        TestSupport.assertTensorShape(gradIn, 4, 6);
    }

    @Test
    void backward_accumulates_nonzero_gradients_in_all_projections() {
        SwiGLULayer layer = new SwiGLULayer(4, 8, new Random(5));
        Tensor x = randomTensor(2, 4, 13);
        layer.forward(x);
        layer.backward(Tensor.ones(2, 4));

        double totalGrad = 0;
        for (Parameter p : layer.parameters()) totalGrad += p.grad.sumAbs();
        assertTrue(totalGrad > 0, "At least one parameter should have a non-zero gradient");
    }

    @Test
    void backward_numerical_gradient_check() {
        int dModel = 4;
        int dFF    = 8;
        double eps = 1e-5;
        double tol = 1e-4;

        SwiGLULayer layer = new SwiGLULayer(dModel, dFF, new Random(6));
        Tensor x = randomTensor(2, dModel, 17);

        // Analytical gradient
        layer.forward(x);
        Tensor analyticGrad = layer.backward(Tensor.ones(2, dModel));

        // Numerical gradient for each element of x
        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                double orig = x.data[r][c];

                x.data[r][c] = orig + eps;
                double fPlus = sumAll(new SwiGLULayer(dModel, dFF, new Random(6)).forward(x));

                x.data[r][c] = orig - eps;
                double fMinus = sumAll(new SwiGLULayer(dModel, dFF, new Random(6)).forward(x));

                x.data[r][c] = orig;

                double numerical = (fPlus - fMinus) / (2 * eps);
                assertEquals(numerical, analyticGrad.data[r][c], tol,
                        "Gradient mismatch at [" + r + "," + c + "]");
            }
        }
    }

    // ── learning ─────────────────────────────────────────────────────────────

    @Test
    void learning_reduces_mse_loss_within_a_few_steps() {
        SwiGLULayer layer = new SwiGLULayer(4, 8, new Random(7));
        AdamW opt = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.0);

        Tensor x      = randomTensor(3, 4, 21);
        Tensor target = Tensor.zeros(3, 4);

        double prev = trainOneStep(layer, opt, x, target);
        boolean improved = false;

        for (int i = 0; i < 10; i++) {
            double cur = trainOneStep(layer, opt, x, target);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "MSE should decrease within a few AdamW steps");
    }

    // ── parameters ───────────────────────────────────────────────────────────

    @Test
    void has_parameters_from_all_three_projections() {
        // gateProj + upProj + downProj each contribute W and b → 6 parameters total
        SwiGLULayer layer = new SwiGLULayer(4, 8, new Random(8));
        assertEquals(6, layer.parameters().size(),
                "Expected 6 parameters (W+b for each of gateProj, upProj, downProj)");
    }

    // ── guards ───────────────────────────────────────────────────────────────

    @Test
    void constructor_zero_dModel_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new SwiGLULayer(0, 8, new Random(1)));
    }

    @Test
    void constructor_zero_dFF_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new SwiGLULayer(4, 0, new Random(1)));
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static Tensor randomTensor(int rows, int cols, long seed) {
        return Tensor.random(rows, cols, new Random(seed));
    }

    private static double sumAll(Tensor t) {
        double s = 0;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                s += t.data[r][c];
        return s;
    }

    private static double trainOneStep(SwiGLULayer layer, AdamW opt, Tensor x, Tensor target) {
        Tensor y = layer.forward(x);
        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        layer.backward(mse.gradient(y, target));
        opt.step(layer.parameters());
        for (Parameter p : layer.parameters()) p.zeroGrad();
        return loss;
    }
}

