package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class LinearTest {

    @Test
    void forwardBackward_shapes_and_basicGradSignals() {
        Linear lin = new Linear(2, 3, new Random(1));

        // overwrite weights for deterministic behavior
        List<Parameter> ps = lin.parameters();
        Parameter W = ps.get(0);
        Parameter b = ps.get(1);

        W.value = Tensor.from2D(new float[][]{
                {1, 0, -1},
                {2, 1,  0}
        });
        b.value = Tensor.from2D(new float[][]{{0.5f, -0.5f, 1.0f}});

        Tensor x = Tensor.from2D(new float[][]{
                {1, 2},
                {-1, 0}
        });

        Tensor y = lin.forward(x);
        TestSupport.assertTensorAllClose(y, Tensor.from2D(new float[][]{
                {5.5f, 1.5f, 0.0f},
                {-0.5f, -0.5f, 2.0f}
        }), 1e-12f);

        Tensor gradOut = Tensor.from2D(new float[][]{
                {1, 1, 1},
                {2, 0, -1}
        });

        Tensor gx = lin.backward(gradOut);
        TestSupport.assertTensorShape(gx, 2, 2);

        // grads exist
        TestSupport.assertTensorShape(W.grad, 2, 3);
        TestSupport.assertTensorShape(b.grad, 1, 3);
    }

    @Test
    void linear_can_reduce_mse_loss_with_optimizer_step() {
        Linear lin = new Linear(2, 3, new Random(1));
        AdamW opt = new AdamW(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Tensor x = Tensor.from2D(new float[][]{
                {1, 2},
                {-1, 0}
        });

        Tensor target = Tensor.from2D(new float[][]{
                {0, 0, 0},
                {0, 0, 0}
        });

        double prev = trainOneStepMSE(lin, opt, x, target);
        boolean improved = false;

        for (int i = 0; i < 10; i++) {
            double cur = trainOneStepMSE(lin, opt, x, target);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "expected loss to decrease within a few steps");
    }

    private static double trainOneStepMSE(Linear lin, AdamW opt, Tensor x, Tensor target) {
        Tensor y = lin.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        lin.backward(gradOut);
        opt.step(lin.parameters());
        for (Parameter p : lin.parameters()) p.zeroGrad();

        return loss;
    }

    // ── finite-difference gradient checks ────────────────────────────────────

    @Test
    void backward_numerical_gradient_check_input_weight_bias() {
        int dIn = 3, dOut = 2;
        float eps = 1e-3f;
        float tol = 5e-3f;

        Linear lin = new Linear(dIn, dOut, new Random(7));
        Tensor x = Tensor.from2D(new float[][]{
                { 0.5f, -1.0f,  2.0f},
                { 1.5f,  0.3f, -0.7f}
        });

        lin.forward(x);
        Tensor dX = lin.backward(Tensor.ones(x.rows, dOut));
        Parameter W = lin.weight();
        Parameter b = lin.bias();
        float[] dW = W.grad.data.clone();
        float[] db = b.grad.data.clone();

        // ∂(Σ out)/∂x
        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < dIn; c++) {
                float orig = x.get(r, c);
                x.set(r, c, orig + eps);
                float fPlus = sumAll(lin.forward(x));
                x.set(r, c, orig - eps);
                float fMinus = sumAll(lin.forward(x));
                x.set(r, c, orig);
                assertEquals((fPlus - fMinus) / (2 * eps), dX.get(r, c), tol,
                        "dX mismatch at [" + r + "," + c + "]");
            }
        }

        // ∂(Σ out)/∂W
        for (int i = 0; i < dIn; i++) {
            for (int j = 0; j < dOut; j++) {
                float orig = W.value.get(i, j);
                W.value.set(i, j, orig + eps);
                float fPlus = sumAll(lin.forward(x));
                W.value.set(i, j, orig - eps);
                float fMinus = sumAll(lin.forward(x));
                W.value.set(i, j, orig);
                assertEquals((fPlus - fMinus) / (2 * eps), dW[i * dOut + j], tol,
                        "dW mismatch at [" + i + "," + j + "]");
            }
        }

        // ∂(Σ out)/∂b
        for (int j = 0; j < dOut; j++) {
            float orig = b.value.get(0, j);
            b.value.set(0, j, orig + eps);
            float fPlus = sumAll(lin.forward(x));
            b.value.set(0, j, orig - eps);
            float fMinus = sumAll(lin.forward(x));
            b.value.set(0, j, orig);
            assertEquals((fPlus - fMinus) / (2 * eps), db[j], tol, "db mismatch at col " + j);
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