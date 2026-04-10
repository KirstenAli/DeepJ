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
}