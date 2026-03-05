package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class MultiHeadSelfAttentionTest {

    @Test
    void constructor_rejectsWhenDModelNotDivisibleByHeads() {
        assertThrows(IllegalArgumentException.class, () -> new MultiHeadSelfAttention(5, 2, true, new Random(1)));
    }

    @Test
    void forward_respectsCausalMask_futureTokensDoNotAffectPastOutputs() {
        int dModel = 4;
        int nHeads = 2;
        int seqLen = 4;

        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(42));

        // Make projections identity so any "leak" would be easy to detect.
        // Parameters() returns [Wq, Wk, Wv, Wo] in that order.
        var ps = attn.parameters();
        assertEquals(4, ps.size());

        Tensor I = Tensor.zeros(dModel, dModel);
        for (int i = 0; i < dModel; i++) I.data[i][i] = 1.0;
        for (Parameter p : ps) {
            p.value = TestSupport.copy(I);
            p.zeroGrad();
        }

        Tensor x1 = TestSupport.tensor(new double[][]{
                { 1,  0,  0,  0},
                { 0,  1,  0,  0},
                { 0,  0,  1,  0},
                { 0,  0,  0,  1}
        });

        Tensor y1 = attn.forward(x1);

        // Modify ONLY the future token (last row) heavily.
        Tensor x2 = TestSupport.copy(x1);
        x2.data[seqLen - 1][0] = 999;
        x2.data[seqLen - 1][1] = 999;
        x2.data[seqLen - 1][2] = 999;
        x2.data[seqLen - 1][3] = 999;

        Tensor y2 = attn.forward(x2);

        // With causal masking, earlier positions must not change.
        for (int r = 0; r < seqLen - 1; r++) {
            for (int c = 0; c < dModel; c++) {
                assertEquals(y1.data[r][c], y2.data[r][c], 1e-7, "past output changed at [" + r + "," + c + "]");
            }
        }
    }

    @Test
    void backward_producesNonZeroGradients_forProjectionMatrices() {
        int dModel = 4;
        int nHeads = 2;
        int seqLen = 3;

        MultiHeadSelfAttention attn = new MultiHeadSelfAttention(dModel, nHeads, true, new Random(7));

        Tensor x = TestSupport.tensor(new double[][]{
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
            assertTrue(TestSupport.tensorSumAbs(p.grad) > 0.0, "Expected non-zero gradient for a projection matrix");
        }
    }
}
