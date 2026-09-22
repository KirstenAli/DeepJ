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
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadSelfAttention(4, 0, true, new Random(1)));
    }

    @Test
    void forward_respectsCausalMask_futureTokensDoNotAffectPastOutputs() {
        int dModel = 4;
        int seqLen = 4;
        MultiHeadSelfAttention attn = identityAttention(dModel);
        Tensor x1 = identityInput(dModel);
        Tensor y1 = attn.forward(x1);
        overwriteLastRow(x1, 999);
        Tensor y2 = attn.forward(x1);
        assertPastRowsEqual(y1, y2, seqLen - 1, dModel);
    }

    private static MultiHeadSelfAttention identityAttention(int dModel) {
        MultiHeadSelfAttention attention = new MultiHeadSelfAttention(
                dModel, 2, true, new Random(42));
        assertEquals(4, attention.parameters().size());
        Tensor identity = Tensor.zeros(dModel, dModel);
        for (int index = 0; index < dModel; index++) {
            identity.data[index * dModel + index] = 1;
        }
        for (Parameter parameter : attention.parameters()) {
            parameter.value = identity;
        }
        return attention;
    }

    private static Tensor identityInput(int size) {
        Tensor input = Tensor.zeros(size, size);
        for (int index = 0; index < size; index++) {
            input.data[index * size + index] = 1;
        }
        return input;
    }

    private static void overwriteLastRow(Tensor tensor, float value) {
        int offset = (tensor.rows - 1) * tensor.cols;
        for (int column = 0; column < tensor.cols; column++) {
            tensor.data[offset + column] = value;
        }
    }

    private static void assertPastRowsEqual(Tensor first, Tensor second, int rows, int columns) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                assertEquals(first.data[r * columns + c], second.data[r * columns + c], 1e-7f,
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

        Tensor x = Tensor.from2D(new float[][]{
                { 0.1f,  0.2f, -0.3f,  0.4f},
                { 0.0f, -0.5f,  0.6f,  0.1f},
                { 0.9f,  0.8f,  0.7f, -0.2f}
        });

        Tensor y = attn.forward(x);
        TestSupport.assertTensorShape(y, seqLen, dModel);

        Tensor gradOut = Tensor.ones(seqLen, dModel);
        Tensor gradIn = attn.backward(gradOut);
        TestSupport.assertTensorShape(gradIn, seqLen, dModel);

        for (Parameter p : attn.parameters()) {
            assertTrue(p.grad.sumAbs() > 0.0f,
                    "Expected non-zero gradient for a projection matrix");
        }
    }

    @Test
    void learning_reduces_mse_loss_within_a_few_steps() {
        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(4, 2, true, new Random(1));
        AdamW opt = new AdamW(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
        Tensor x = Tensor.from2D(new float[][]{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}});
        Tensor target = Tensor.from2D(new float[][]{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}});
        TestSupport.assertLossDecreases(() -> trainOneStepMSE(attn, opt, x, target), 10,
                "expected loss to decrease within a few optimizer steps");
    }

    private static double trainOneStepMSE(MultiHeadSelfAttention attn, AdamW opt, Tensor x, Tensor target) {
        Tensor y = attn.forward(x);

        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        attn.backward(gradOut);
        opt.step(attn.parameters());
        for (Parameter p : attn.parameters()) {
            p.zeroGrad();
        }

        return loss;
    }

    @Test
    void backward_matchesNumericalInputGradient() {
        int dModel = 4, nHeads = 2, seqLen = 3;
        float eps = 2e-3f, tol = 1e-2f;

        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(7));
        Tensor x = Tensor.random(seqLen, dModel, new Random(99));

        attn.forward(x);
        Tensor dX = attn.backward(Tensor.ones(seqLen, dModel));
        assertInputGradients(attn, x, dX, eps, tol);
    }

    private static void assertInputGradients(MultiHeadSelfAttention attn, Tensor x,
                                             Tensor dX, float eps, float tol) {
        for (int r = 0; r < x.rows; r++) {
            for (int c = 0; c < x.cols; c++) {
                float orig = x.get(r, c);
                x.set(r, c, orig + eps);
                float fPlus = sumAll(attn.forward(x));
                x.set(r, c, orig - eps);
                float fMinus = sumAll(attn.forward(x));
                x.set(r, c, orig);
                assertEquals((fPlus - fMinus) / (2 * eps), dX.get(r, c), tol,
                        "input grad mismatch at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backward_matchesNumericalOutputWeightGradient() {
        int dModel = 4, nHeads = 2, seqLen = 3;
        float eps = 2e-3f, tol = 1e-2f;
        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(11));
        Tensor x = Tensor.random(seqLen, dModel, new Random(31));
        attn.forward(x);
        attn.backward(Tensor.ones(seqLen, dModel));
        Parameter Wo = attn.parameters().get(3);
        float[] dWo = Wo.grad.data.clone();
        assertOutputWeightGradient(attn, x, Wo, dWo, dModel, eps, tol);
    }

    private static void assertOutputWeightGradient(MultiHeadSelfAttention attn, Tensor x,
                                                   Parameter weight, float[] gradient,
                                                   int dModel, float eps, float tol) {
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                float orig = weight.value.get(i, j);
                weight.value.set(i, j, orig + eps);
                float fPlus = sumAll(attn.forward(x));
                weight.value.set(i, j, orig - eps);
                float fMinus = sumAll(attn.forward(x));
                weight.value.set(i, j, orig);
                assertEquals((fPlus - fMinus) / (2 * eps), gradient[i * dModel + j], tol,
                        "dWo mismatch at [" + i + "," + j + "]");
            }
        }
    }

    private static float sumAll(Tensor t) {
        float s = 0.0f;
        for (int r = 0; r < t.rows; r++) {
            for (int c = 0; c < t.cols; c++) {
                s += t.data[r * t.cols + c];
            }
        }
        return s;
    }
}
