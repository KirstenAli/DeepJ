package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.activations.ReLU;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class FNNTest {

    @Test
    void fnn_reduces_mse_loss_within_a_few_steps() {
        FNN mlp = new FNN(3, new int[]{4}, 2, ReLU::new, new Random(123));
        AdamW opt = new AdamW(0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Tensor x = Tensor.from2D(new float[][]{
                { 1.0f,  0.0f, -1.0f},
                { 0.5f,  2.0f,  1.0f}
        });

        Tensor target = Tensor.from2D(new float[][]{
                { 1.0f,  0.0f},
                { 0.0f,  1.0f}
        });

        double prev = trainOneStepMSE(mlp, opt, x, target);
        boolean improved = false;

        for (int i = 0; i < 20; i++) {
            double cur = trainOneStepMSE(mlp, opt, x, target);
            if (cur < prev) {
                improved = true;
                break;
            }
            prev = cur;
        }

        assertTrue(improved, "expected MSE loss to decrease within a few optimizer steps");
    }

    private static double trainOneStepMSE(FNN mlp, AdamW opt, Tensor x, Tensor target) {
        Tensor y = mlp.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        mlp.backward(gradOut);
        opt.step(mlp.parameters());

        for (Parameter p : mlp.parameters()) p.zeroGrad();

        return loss;
    }
}