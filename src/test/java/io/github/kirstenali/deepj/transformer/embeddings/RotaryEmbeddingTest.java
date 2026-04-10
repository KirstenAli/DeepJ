package io.github.kirstenali.deepj.transformer.embeddings;

import io.github.kirstenali.deepj.TestSupport;
import io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.attention.RoPEMultiHeadSelfAttention;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RotaryEmbedding} and its integration with
 * {@link RoPEMultiHeadSelfAttention} and {@link LlamaTransformerBlock}.
 *
 * <p>Covers:
 * <ul>
 *   <li>Cos/sin table values at known positions</li>
 *   <li>Shape preservation in apply() and applyBackward()</li>
 *   <li>Rotation is norm-preserving (orthogonal transformation)</li>
 *   <li>apply + applyBackward is the identity (Rᵀ·R = I)</li>
 *   <li>Position 0 leaves vector unchanged (cos=1, sin=0)</li>
 *   <li>MHSA with RoPE: forward/backward shape contracts</li>
 *   <li>MHSA with RoPE: gradients are non-zero</li>
 *   <li>Llama-style GPTTransformerBlock (RMSNorm + RoPE MHSA + SwiGLU): shape + learning</li>
 *   <li>Guards: odd headDim throws, seqLen overflow throws</li>
 * </ul>
 */
class RotaryEmbeddingTest {

    // ── table values ─────────────────────────────────────────────────────────

    @Test
    void cos_sin_at_position_zero_are_one_and_zero() {
        // θ_{0,i} = 0 for all i → cos = 1, sin = 0
        RotaryEmbedding rope = new RotaryEmbedding(4, 8);
        // Apply a single-head, single-position tensor [1 x 4] at pos=0; result should equal input.
        Tensor x = Tensor.from2D(new float[][]{{1.0f, 2.0f, 3.0f, 4.0f}});
        Tensor y = rope.apply(x, 1, 1);
        TestSupport.assertTensorAllClose(x, y, 1e-6f);
    }

    @Test
    void cos_sin_first_dim_at_position_one_are_cos1_sin1() {
        // For dim i=0: θ = 1 / 10000^0 = 1  →  cos(1), sin(1)
        RotaryEmbedding rope = new RotaryEmbedding(4, 8);
        // Single head, two positions (pos 0 and pos 1), headDim=4
        Tensor x = Tensor.from2D(new float[][]{{1.0f, 0.0f, 0.0f, 0.0f},   // pos 0
                                             {1.0f, 0.0f, 0.0f, 0.0f}});  // pos 1
        Tensor y = rope.apply(x, 2, 1);

        // pos 0: no rotation
        assertEquals(1.0f, y.data[0], 1e-6f);
        assertEquals(0.0f, y.data[1], 1e-6f);

        // pos 1: rotated by θ = 1 → x_rot[0] = cos(1), x_rot[1] = sin(1)
        assertEquals(Math.cos(1.0f), y.data[1 * 4 + 0], 1e-6f);
        assertEquals(Math.sin(1.0f), y.data[1 * 4 + 1], 1e-6f);
    }

    // ── shape ────────────────────────────────────────────────────────────────

    @Test
    void apply_preserves_shape() {
        RotaryEmbedding rope = new RotaryEmbedding(8, 16);
        Tensor x = Tensor.random(3 * 8, 8, new Random(1)); // nHeads=3, seqLen=8, headDim=8
        Tensor y = rope.apply(x, 8, 3);
        TestSupport.assertTensorShape(y, 3 * 8, 8);
    }

    @Test
    void applyBackward_preserves_shape() {
        RotaryEmbedding rope = new RotaryEmbedding(8, 16);
        Tensor x = Tensor.random(2 * 4, 8, new Random(2));
        Tensor y = rope.applyBackward(x, 4, 2);
        TestSupport.assertTensorShape(y, 2 * 4, 8);
    }

    // ── mathematical properties ───────────────────────────────────────────────

    @Test
    void rotation_is_norm_preserving() {
        RotaryEmbedding rope = new RotaryEmbedding(4, 16);
        Tensor x = Tensor.from2D(new float[][]{{3.0f, 4.0f, 1.0f, 2.0f},   // pos 0
                                             {1.0f, 1.0f, 1.0f, 1.0f},   // pos 1
                                             {2.0f, -3.0f, 0.0f, 5.0f}}); // pos 2  (nHeads=1, seqLen=3)
        Tensor y = rope.apply(x, 3, 1);

        for (int r = 0; r < x.rows; r++) {
            double normX = rowNorm(x, r);
            double normY = rowNorm(y, r);
            assertEquals(normX, normY, 1e-6f, "Norm should be preserved at row " + r);
        }
    }

    @Test
    void apply_then_applyBackward_is_identity() {
        // R⁻¹ · R x = x  (rotation matrix is orthogonal)
        RotaryEmbedding rope = new RotaryEmbedding(8, 32);
        Tensor x = Tensor.random(2 * 5, 8, new Random(3));  // nHeads=2, seqLen=5
        Tensor rotated   = rope.apply(x, 5, 2);
        Tensor recovered = rope.applyBackward(rotated, 5, 2);
        TestSupport.assertTensorAllClose(x, recovered, 1e-6f);
    }

    @Test
    void applyBackward_then_apply_is_identity() {
        // R · R⁻¹ x = x
        RotaryEmbedding rope = new RotaryEmbedding(4, 16);
        Tensor x  = Tensor.random(3 * 4, 4, new Random(4));
        Tensor inv = rope.applyBackward(x, 4, 3);
        Tensor back = rope.apply(inv, 4, 3);
        TestSupport.assertTensorAllClose(x, back, 1e-6f);
    }

    @Test
    void multi_head_rotation_each_head_sees_same_positions() {
        // Two heads, same input per head → both should be rotated identically.
        RotaryEmbedding rope = new RotaryEmbedding(4, 8);
        // Build a 2-head tensor where both heads have the same content.
        float[][] rowData = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
        Tensor xSingle = Tensor.from2D(rowData);              // 1 head, seqLen=2
        Tensor xDouble = Tensor.from2D(new float[][]{        // 2 heads, seqLen=2
                rowData[0], rowData[1],  // head 0
                rowData[0], rowData[1]   // head 1 (same)
        });

        Tensor ySingle = rope.apply(xSingle, 2, 1);
        Tensor yDouble = rope.apply(xDouble, 2, 2);

        // Head 0 and head 1 of yDouble should match ySingle.
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 4; c++) {
                assertEquals(ySingle.data[r * 4 + c], yDouble.data[r * 4 + c],       1e-6f, "head0 row " + r);
                assertEquals(ySingle.data[r * 4 + c], yDouble.data[(r + 2) * 4 + c], 1e-6f, "head1 row " + r);
            }
        }
    }

    // ── guards ───────────────────────────────────────────────────────────────

    @Test
    void odd_headDim_throws() {
        assertThrows(IllegalArgumentException.class, () -> new RotaryEmbedding(3, 16));
    }

    @Test
    void zero_headDim_throws() {
        assertThrows(IllegalArgumentException.class, () -> new RotaryEmbedding(0, 16));
    }

    @Test
    void seqLen_exceeds_maxSeqLen_throws() {
        RotaryEmbedding rope = new RotaryEmbedding(4, 8);
        Tensor x = Tensor.random(1 * 9, 4, new Random(1));
        assertThrows(IllegalArgumentException.class, () -> rope.apply(x, 9, 1));
    }

    // ── MHSA integration ─────────────────────────────────────────────────────

    @Test
    void mhsa_with_rope_forward_returns_correct_shape() {
        int dModel = 8, nHeads = 2, seqLen = 4;
        RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, 16);
        RoPEMultiHeadSelfAttention attn = new RoPEMultiHeadSelfAttention(dModel, nHeads, true, rope, new Random(10));

        Tensor x = Tensor.random(seqLen, dModel, new Random(11));
        Tensor y = attn.forward(x);
        TestSupport.assertTensorShape(y, seqLen, dModel);
    }

    @Test
    void mhsa_with_rope_backward_returns_correct_shape() {
        int dModel = 8, nHeads = 2, seqLen = 4;
        RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, 16);
        RoPEMultiHeadSelfAttention attn = new RoPEMultiHeadSelfAttention(dModel, nHeads, true, rope, new Random(12));

        Tensor x = Tensor.random(seqLen, dModel, new Random(13));
        attn.forward(x);
        Tensor gradIn = attn.backward(Tensor.ones(seqLen, dModel));
        TestSupport.assertTensorShape(gradIn, seqLen, dModel);
    }

    @Test
    void mhsa_with_rope_accumulates_nonzero_weight_gradients() {
        int dModel = 8, nHeads = 2, seqLen = 3;
        RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, 16);
        RoPEMultiHeadSelfAttention attn = new RoPEMultiHeadSelfAttention(dModel, nHeads, true, rope, new Random(14));

        attn.forward(Tensor.random(seqLen, dModel, new Random(15)));
        attn.backward(Tensor.ones(seqLen, dModel));

        double totalGrad = 0;
        for (Parameter p : attn.parameters()) totalGrad += p.grad.sumAbs();
        assertTrue(totalGrad > 0, "RoPE-enabled MHSA should produce non-zero weight gradients");
    }

    // ── Llama-style block integration ─────────────────────────────────────────

    @Test
    void llama_style_block_forward_returns_correct_shape() {
        int dModel = 8, nHeads = 2, dFF = 16, seqLen = 4;
        LlamaTransformerBlock block = llamaBlock(dModel, nHeads, dFF, 1);

        Tensor x = Tensor.random(seqLen, dModel, new Random(20));
        Tensor y = block.forward(x);
        TestSupport.assertTensorShape(y, seqLen, dModel);
    }

    @Test
    void llama_style_block_backward_returns_correct_shape() {
        int dModel = 8, nHeads = 2, dFF = 16, seqLen = 4;
        LlamaTransformerBlock block = llamaBlock(dModel, nHeads, dFF, 2);

        Tensor x = Tensor.random(seqLen, dModel, new Random(21));
        block.forward(x);
        Tensor gradIn = block.backward(Tensor.ones(seqLen, dModel));
        TestSupport.assertTensorShape(gradIn, seqLen, dModel);
    }

    @Test
    void llama_style_block_learning_reduces_mse() {
        int dModel = 8, nHeads = 2, dFF = 16, seqLen = 4;
        LlamaTransformerBlock block = llamaBlock(dModel, nHeads, dFF, 3);
        AdamW opt = new AdamW(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Tensor x      = Tensor.random(seqLen, dModel, new Random(22));
        Tensor target = Tensor.zeros(seqLen, dModel);

        double prev = trainOneStep(block, opt, x, target);
        boolean improved = false;

        for (int i = 0; i < 10; i++) {
            double cur = trainOneStep(block, opt, x, target);
            if (cur < prev) { improved = true; break; }
            prev = cur;
        }

        assertTrue(improved, "Llama-style block MSE should decrease within a few steps");
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static LlamaTransformerBlock llamaBlock(int dModel, int nHeads, int dFF, long seed) {
        return new LlamaTransformerBlock(dModel, nHeads, dFF, 64, new Random(seed));
    }

    private static double trainOneStep(LlamaTransformerBlock block, AdamW opt, Tensor x, Tensor target) {
        Tensor y = block.forward(x);
        MSELoss mse = new MSELoss();
        double loss = mse.loss(y, target);
        block.backward(mse.gradient(y, target));
        opt.step(block.parameters());
        for (Parameter p : block.parameters()) p.zeroGrad();
        return loss;
    }

    private static double rowNorm(Tensor t, int row) {
        double s = 0;
        int base = row * t.cols;
        for (int c = 0; c < t.cols; c++) s += t.data[base + c] * t.data[base + c];
        return Math.sqrt(s);
    }
}

