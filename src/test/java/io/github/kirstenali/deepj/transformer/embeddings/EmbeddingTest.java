package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class EmbeddingTest {

    @Test
    void forward_pullsCorrectRows_fromWeightMatrix() {
        Embedding emb = new Embedding(5, 3, new Random(1));

        for (int i = 0; i < 5; i++) {
            emb.weight().value.data[i * 3 + 0] = i;
            emb.weight().value.data[i * 3 + 1] = i + 1;
            emb.weight().value.data[i * 3 + 2] = i + 2;
        }

        Tensor out = emb.forward(new int[]{3, 1, 3});
        TestSupport.assertTensorShape(out, 3, 3);

        assertArrayEquals(new double[]{3, 4, 5}, out.rowData(0), 1e-12);
        assertArrayEquals(new double[]{1, 2, 3}, out.rowData(1), 1e-12);
        assertArrayEquals(new double[]{3, 4, 5}, out.rowData(2), 1e-12);
    }

    @Test
    void backward_accumulatesGradients_forRepeatedTokenIds() {
        Embedding emb = new Embedding(4, 2, new Random(2));
        emb.weight().zeroGrad();

        emb.forward(new int[]{1, 1, 3});

        Tensor gradOut = new Tensor(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        });
        emb.backward(gradOut);

        // id=1 appears twice => grads sum
        assertEquals(1.0 + 3.0, emb.weight().grad.data[1 * 2 + 0], 1e-12);
        assertEquals(2.0 + 4.0, emb.weight().grad.data[1 * 2 + 1], 1e-12);

        // id=3 appears once
        assertEquals(5.0, emb.weight().grad.data[3 * 2 + 0], 1e-12);
        assertEquals(6.0, emb.weight().grad.data[3 * 2 + 1], 1e-12);
    }

    @Test
    void forward_rejectsOutOfRangeIds() {
        Embedding emb = new Embedding(3, 2, new Random(3));
        assertThrows(IllegalArgumentException.class, () -> emb.forward(new int[]{-1}));
        assertThrows(IllegalArgumentException.class, () -> emb.forward(new int[]{3}));
    }

    @Test
    void embedding_row_can_learn_target_vector_via_sgd_mse_decreases() {
        Embedding emb = new Embedding(6, 3, new Random(4));

        int id = 2;
        Tensor target = new Tensor(new double[][]{
                { 0.25, -0.50, 1.25 }
        });

        double lr = 0.1;

        double prev = oneSgdStepMSE(emb, id, target, lr);
        boolean improved = false;

        // Require improvement within a few steps (robust against minor numerical wiggles)
        for (int i = 0; i < 20; i++) {
            double cur = oneSgdStepMSE(emb, id, target, lr);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "expected MSE to decrease after SGD updates on the embedding row");
    }

    private static double oneSgdStepMSE(Embedding emb, int id, Tensor target, double lr) {
        emb.weight().zeroGrad();

        // forward on a single token id -> output shape (1, d)
        Tensor y = emb.forward(new int[]{id});
        TestSupport.assertTensorShape(y, 1, target.cols);

        // MSE loss + grad wrt output
        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        emb.backward(gradOut);  // or pe.back

        // manual SGD update: only row "id" needs updating, but updating whole matrix is fine too.
        Tensor W = emb.weight().value;
        Tensor dW = emb.weight().grad;

        for (int c = 0; c < W.cols; c++) {
            W.data[id * W.cols + c] -= lr * dW.data[id * dW.cols + c];
        }

        return loss;
    }
}
