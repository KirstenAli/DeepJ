package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class MultiHeadSelfAttentionTest {

    @Test
    void constructor_rejectsWhenDModelNotDivisibleByHeads() {
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadSelfAttention(5, 2, true, new Random(1)));
    }

    @Test
    void forward_respectsCausalMask_futureTokensDoNotAffectPastOutputs() {
        int dModel = 4;
        int nHeads = 2;
        int seqLen = 4;

        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(42));

        // Make projections identity so any "leak" is easy to detect.
        // parameters() returns [Wq, Wk, Wv, Wo]
        var ps = attn.parameters();
        assertEquals(4, ps.size());

        Tensor I = Tensor.zeros(dModel, dModel);
        for (int i = 0; i < dModel; i++) I.data[i * dModel + i] = 1.0f;

        for (Parameter p : ps) {
            p.value = I;
            p.zeroGrad();
        }

        Tensor x1 = Tensor.from2D(new double[][]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        });

        Tensor y1 = attn.forward(x1);

        // Modify ONLY the future token (last row)
        x1.data[(seqLen - 1) * dModel + 0] = 999;
        x1.data[(seqLen - 1) * dModel + 1] = 999;
        x1.data[(seqLen - 1) * dModel + 2] = 999;
        x1.data[(seqLen - 1) * dModel + 3] = 999;

        Tensor y2 = attn.forward(x1);

        for (int r = 0; r < seqLen - 1; r++) {
            for (int c = 0; c < dModel; c++) {
                assertEquals(y1.data[r * dModel + c], y2.data[r * dModel + c], 1e-7,
                        "past output changed at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backward_producesNonZeroGradients_forProjectionMatrices() {
        int dModel = 4;
        int nHeads = 2;
        int seqLen = 3;

        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(7));

        Tensor x = Tensor.from2D(new double[][]{
                { 0.1,  0.2, -0.3,  0.4},
                { 0.0, -0.5,  0.6,  0.1},
                { 0.9,  0.8,  0.7, -0.2}
        });

        Tensor y = attn.forward(x);
        TestSupport.assertTensorShape(y, seqLen, dModel);

        Tensor gradOut = Tensor.ones(seqLen, dModel);
        Tensor gradIn = attn.backward(gradOut);
        TestSupport.assertTensorShape(gradIn, seqLen, dModel);

        for (Parameter p : attn.parameters()) {
            assertTrue(p.grad.sumAbs() > 0.0,
                    "Expected non-zero gradient for a projection matrix");
        }
    }

    @Test
    void learning_reduces_mse_loss_within_a_few_steps() {
        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(4, 2, true, new Random(1));
        AdamW opt = new AdamW(0.01, 0.9, 0.999, 1e-8, 0.0);

        Tensor x = Tensor.from2D(new double[][]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}
        });

        Tensor target = Tensor.from2D(new double[][]{
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        });

        double prev = trainOneStepMSE(attn, opt, x, target);
        boolean improved = false;

        // AdamW may not strictly improve every single step due to momentum;
        // require improvement within a few steps.
        for (int i = 0; i < 10; i++) {
            double cur = trainOneStepMSE(attn, opt, x, target);
            if (cur < prev) {
                improved = true;
                break;
            }
            prev = cur;
        }

        assertTrue(improved, "expected loss to decrease within a few optimizer steps");
    }

    private static double trainOneStepMSE(MultiHeadSelfAttention attn, AdamW opt, Tensor x, Tensor target) {
        Tensor y = attn.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        attn.backward(gradOut);
        opt.step(attn.parameters());
        for (Parameter p : attn.parameters()) p.zeroGrad();

        return loss;
    }
}