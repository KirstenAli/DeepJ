package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;

import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LayerNorm1DTest {

    @Test
    void gammaBeta_update_can_reduce_mse_loss_within_a_few_steps() {
        LayerNorm1D ln = new LayerNorm1D(3);
        AdamW opt = new AdamW(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Tensor x = Tensor.from2D(new float[][]{
                { 1.0f,  2.0f,  3.0f},
                { 2.0f,  0.0f, -2.0f}
        });

        Tensor target = Tensor.from2D(new float[][]{
                { 0.5f,  0.5f,  0.5f},
                {-0.5f, -0.5f, -0.5f}
        });

        double prev = oneStepMSE(ln, opt, x, target);
        boolean improved = false;

        for (int i = 0; i < 10; i++) {
            double cur = oneStepMSE(ln, opt, x, target);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "expected loss to decrease within a few optimizer steps");
    }

    private static double oneStepMSE(LayerNorm1D ln, AdamW opt, Tensor x, Tensor target) {
        Tensor y = ln.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        ln.backward(gradOut);
        opt.step(ln.parameters());
        for (Parameter p : ln.parameters()) p.zeroGrad();

        return loss;
    }

    // ── finite-difference gradient checks ────────────────────────────────────

    @Test
    void backward_numerical_gradient_check_wrt_input() {
        int dim = 3;
        float eps = 1e-3f;
        float tol = 3e-3f;

        LayerNorm1D ln = new LayerNorm1D(dim);
        Tensor x = Tensor.from2D(new float[][]{
                { 0.5f, -1.0f,  2.0f},
                { 1.5f,  0.3f, -0.7f}
        });

        ln.forward(x);
        Tensor analytic = ln.backward(Tensor.ones(x.rows, dim));

        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < dim; c++) {
                float orig = x.get(r, c);
                x.set(r, c, orig + eps);
                float fPlus = sumAll(ln.forward(x));
                x.set(r, c, orig - eps);
                float fMinus = sumAll(ln.forward(x));
                x.set(r, c, orig);
                float numerical = (fPlus - fMinus) / (2 * eps);
                assertEquals(numerical, analytic.get(r, c), tol,
                        "input grad mismatch at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backward_numerical_gradient_check_wrt_gamma() {
        int dim = 3;
        float eps = 1e-3f;
        float tol = 3e-3f;

        LayerNorm1D ln = new LayerNorm1D(dim);
        Tensor x = Tensor.from2D(new float[][]{
                { 0.5f, -1.0f,  2.0f},
                { 1.5f,  0.3f, -0.7f}
        });

        ln.forward(x);
        ln.backward(Tensor.ones(x.rows, dim));
        Parameter gamma = ln.parameters().get(0);
        float[] analytic = gamma.grad.data.clone();

        for (int c = 0; c < dim; c++) {
            float orig = gamma.value.get(0, c);
            gamma.value.set(0, c, orig + eps);
            float fPlus = sumAll(ln.forward(x));
            gamma.value.set(0, c, orig - eps);
            float fMinus = sumAll(ln.forward(x));
            gamma.value.set(0, c, orig);
            float numerical = (fPlus - fMinus) / (2 * eps);
            assertEquals(numerical, analytic[c], tol, "gamma grad mismatch at col " + c);
        }
    }

    @Test
    void backward_beta_gradient_equals_summed_upstream() {
        int dim = 3;
        LayerNorm1D ln = new LayerNorm1D(dim);
        Tensor x = Tensor.from2D(new float[][]{
                { 0.5f, -1.0f,  2.0f},
                { 1.5f,  0.3f, -0.7f}
        });

        ln.forward(x);
        ln.backward(Tensor.ones(x.rows, dim));

        // out = xHat·gamma + beta (broadcast over rows) → dL/dbeta[c] = Σ_r gradOut[r,c] = rows
        Parameter beta = ln.parameters().get(1);
        for (int c = 0; c < dim; c++) {
            assertEquals((float) x.rows, beta.grad.data[c], 1e-6f, "beta grad at col " + c);
        }
    }

    private static float sumAll(Tensor t) {
        float s = 0.0f;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                s += t.data[r * t.cols + c];
        return s;
    }
}