package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TransformerBlockTest {

    @Test
    void forward_isIdentity_whenAllTrainableParametersAreZero() {
        int dModel = 4;
        TransformerBlock block = new TransformerBlock(dModel, 2, 8, new Random(1));

        // Zero ALL trainable params. With residual connections, this should yield y == x.
        for (Parameter p : block.parameters()) {
            p.value = Tensor.zeros(p.value.rows, p.value.cols);
            p.zeroGrad();
        }

        Tensor x = new Tensor(new double[][]{
                { 1,  2,  3,  4},
                {-1, -2, -3, -4}
        });

        Tensor y = block.forward(x);
        TestSupport.assertTensorAllClose(x, y, 1e-12);
    }

    @Test
    void backward_returnsSameShape_andAccumulatesSomeGradients() {
        TransformerBlock block = new TransformerBlock(4, 2, 8, new Random(2));

        Tensor x = new Tensor(new double[][]{
                { 0.2, -0.1,  0.3,  0.0},
                { 0.0,  0.4, -0.2,  0.1},
                {-0.3,  0.2,  0.1, -0.4}
        });

        Tensor y = block.forward(x);
        Tensor gradOut = Tensor.ones(y.rows, y.cols);
        Tensor gradIn = block.backward(gradOut);

        TestSupport.assertTensorShape(gradIn, x.rows, x.cols);

        // Not every param is guaranteed to be non-zero, but at least one should be.
        double totalGrad = 0.0;
        for (Parameter p : block.parameters()) totalGrad += p.grad.sumAbs();
        assertTrue(totalGrad > 0.0, "Expected some non-zero gradients in block parameters");
    }

    @Test
    void learning_can_reduce_mse_loss_within_a_few_steps() {
        TransformerBlock block = new TransformerBlock(4, 2, 8, new Random(3));
        AdamW opt = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.0);

        Tensor x = new Tensor(new double[][]{
                { 0.2, -0.1,  0.3,  0.0},
                { 0.0,  0.4, -0.2,  0.1},
                {-0.3,  0.2,  0.1, -0.4}
        });

        // Simple deterministic target: all zeros (same shape).
        Tensor target = Tensor.zeros(x.rows, x.cols);

        double prev = trainOneStepMSE(block, opt, x, target);
        boolean improved = false;

        // AdamW may not improve every single step due to momentum/bias correction,
        // so we require improvement within a handful of steps.
        for (int i = 0; i < 10; i++) {
            double cur = trainOneStepMSE(block, opt, x, target);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "Expected MSE loss to decrease within a few optimizer steps");
    }

    private static double trainOneStepMSE(TransformerBlock block, AdamW opt, Tensor x, Tensor target) {
        Tensor y = block.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        block.backward(gradOut);
        opt.step(block.parameters());
        for (Parameter p : block.parameters()) p.zeroGrad();

        return loss;
    }
}