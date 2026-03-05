package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class PositionalEmbeddingTest {

    @Test
    void forward_returnsFirstSeqLenRows() {
        PositionalEmbedding pe = new PositionalEmbedding(4, 2, new Random(1));

        // deterministic: row i = [10+i, 20+i]
        var p = pe.parameters().get(0);
        for (int i = 0; i < 4; i++) {
            p.value.data[i][0] = 10 + i;
            p.value.data[i][1] = 20 + i;
        }

        Tensor out = pe.forward(3);
        TestSupport.assertTensorShape(out, 3, 2);
        assertArrayEquals(new double[]{10, 20}, out.data[0], 1e-12);
        assertArrayEquals(new double[]{11, 21}, out.data[1], 1e-12);
        assertArrayEquals(new double[]{12, 22}, out.data[2], 1e-12);
    }

    @Test
    void backward_accumulatesOnlyFirstSeqLenRows() {
        PositionalEmbedding pe = new PositionalEmbedding(5, 3, new Random(2));
        var w = pe.parameters().get(0);
        w.zeroGrad();

        Tensor gradOut = Tensor.ones(2, 3); // seqLen = 2
        pe.backward(gradOut);

        // first two rows should be updated, later rows untouched
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 3; c++) {
                double expected = (r < 2) ? 1.0 : 0.0;
                assertEquals(expected, w.grad.data[r][c], 1e-12, "grad mismatch at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void forward_rejectsSeqLenExceedingMax() {
        PositionalEmbedding pe = new PositionalEmbedding(2, 4, new Random(3));
        assertThrows(IllegalArgumentException.class, () -> pe.forward(3));
    }

    @Test
    void positionalEmbedding_can_learn_target_rows_via_sgd_mse_decreases() {
        PositionalEmbedding pe = new PositionalEmbedding(6, 3, new Random(4));
        var w = pe.parameters().get(0);

        int seqLen = 3;

        // target for first seqLen rows (3x3)
        Tensor target = new Tensor(new double[][]{
                { 0.5,  0.0, -0.5},
                { 1.0,  1.0,  1.0},
                {-1.0,  0.5,  2.0}
        });

        double lr = 0.1;

        double prev = oneSgdStepMSE(pe, w, seqLen, target, lr);
        boolean improved = false;

        for (int i = 0; i < 30; i++) {
            double cur = oneSgdStepMSE(pe, w, seqLen, target, lr);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "expected MSE to decrease after SGD updates on positional rows");
    }

    private static double oneSgdStepMSE(PositionalEmbedding pe,
                                        Parameter w,
                                        int seqLen,
                                        Tensor target,
                                        double lr) {
        w.zeroGrad();

        // forward
        Tensor y = pe.forward(seqLen);
        TestSupport.assertTensorShape(y, seqLen, target.cols);

        // MSE loss + grad wrt output
        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        Tensor gradOut = mse.gradient(y, target);

        pe.backward(gradOut);

        // manual SGD update: only first seqLen rows should update
        for (int r = 0; r < seqLen; r++) {
            for (int c = 0; c < w.value.cols; c++) {
                w.value.data[r][c] -= lr * w.grad.data[r][c];
            }
        }

        return loss;
    }
}
