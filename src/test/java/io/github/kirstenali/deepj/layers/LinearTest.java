package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class LinearTest {

    @Test
    void forwardBackward_shapes_and_basicGradSignals() {
        Linear lin = new Linear(2, 3, new Random(1));

        // overwrite weights for deterministic behavior
        List<Parameter> ps = lin.parameters();
        Parameter W = ps.get(0);
        Parameter b = ps.get(1);

        W.value = new Tensor(new double[][]{
                {1, 0, -1},
                {2, 1,  0}
        });
        b.value = new Tensor(new double[][]{{0.5, -0.5, 1.0}});

        Tensor x = new Tensor(new double[][]{
                {1, 2},
                {-1, 0}
        });

        Tensor y = lin.forward(x);
        TestSupport.assertTensorAllClose(y, new Tensor(new double[][]{
                {5.5, 1.5, 0.0},
                {-0.5, -0.5, 2.0}
        }), 1e-12);

        Tensor gradOut = new Tensor(new double[][]{
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
        AdamW opt = new AdamW(0.05, 0.9, 0.999, 1e-8, 0.0);

        Tensor x = new Tensor(new double[][]{
                {1, 2},
                {-1, 0}
        });

        Tensor target = new Tensor(new double[][]{
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
}