package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class MultiHeadLatentAttentionTest {

    private static final int D_MODEL  = 32;
    private static final int N_HEADS  = 4;
    private static final int Q_RANK   = 16;
    private static final int KV_RANK  = 8;
    private static final int SEQ_LEN  = 6;
    private static final int MAX_SEQ  = 16;

    private MultiHeadLatentAttention attn;

    @BeforeEach
    void setUp() {
        RotaryEmbedding rope = new RotaryEmbedding(D_MODEL / N_HEADS, MAX_SEQ);
        attn = new MultiHeadLatentAttention(D_MODEL, N_HEADS, Q_RANK, KV_RANK, rope, new Random(1L));
    }

    // ── constructor ────────────────────────────────────────────────

    @Test
    void constructor_rejectsDModelNotDivisibleByNHeads() {
        RotaryEmbedding rope = new RotaryEmbedding(8, MAX_SEQ);
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadLatentAttention(33, N_HEADS, Q_RANK, KV_RANK, rope, new Random()));
    }

    @Test
    void constructor_rejectsNonPositiveRanks() {
        RotaryEmbedding rope = new RotaryEmbedding(D_MODEL / N_HEADS, MAX_SEQ);
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadLatentAttention(D_MODEL, N_HEADS, 0, KV_RANK, rope, new Random()));
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadLatentAttention(D_MODEL, N_HEADS, Q_RANK, 0, rope, new Random()));
    }

    // ── forward ────────────────────────────────────────────────────

    @Test
    void forward_outputShapeMatchesInput() {
        Tensor x = Tensor.random(SEQ_LEN, D_MODEL, new Random(2L));
        Tensor out = attn.forward(x);

        assertEquals(SEQ_LEN, out.rows);
        assertEquals(D_MODEL, out.cols);
    }

    @Test
    void forward_singleToken_doesNotThrow() {
        Tensor x = Tensor.random(1, D_MODEL, new Random(3L));
        assertDoesNotThrow(() -> attn.forward(x));
    }

    // ── parameters ─────────────────────────────────────────────────

    @Test
    void parameters_sixTotal() {
        // Wdq, Wuq, Wdkv, Wuk, Wuv, Wo
        assertEquals(6, attn.parameters().size());
    }

    // ── backward ───────────────────────────────────────────────────

    @Test
    void backward_accumulatesGradients() {
        Tensor x   = Tensor.random(SEQ_LEN, D_MODEL, new Random(4L));
        Tensor out = attn.forward(x);
        Tensor dOut = Tensor.ones(out.rows, out.cols);

        attn.parameters().forEach(p -> p.zeroGrad());
        Tensor dX = attn.backward(dOut);

        // input gradient shape
        assertEquals(SEQ_LEN, dX.rows);
        assertEquals(D_MODEL, dX.cols);

        // at least one parameter gradient must be non-zero
        boolean anyNonZero = attn.parameters().stream()
                .anyMatch(p -> p.grad.sumAbs() > 0.0f);
        assertTrue(anyNonZero, "at least one grad must be non-zero after backward");
    }

    @Test
    void backward_inputGradientIsNonZero() {
        Tensor x   = Tensor.random(SEQ_LEN, D_MODEL, new Random(5L));
        Tensor out = attn.forward(x);
        Tensor dX  = attn.backward(Tensor.ones(out.rows, out.cols));

        assertTrue(dX.sumAbs() > 0.0f, "input gradient must be non-zero");
    }

    @Test
    void backward_matchesNumericalInputGradient() {
        float eps = 2e-3f, tol = 1.5e-2f;
        Tensor x = Tensor.random(SEQ_LEN, D_MODEL, new Random(6L));

        attn.forward(x);
        Tensor dX = attn.backward(Tensor.ones(SEQ_LEN, D_MODEL));

        for (int r = 0; r < SEQ_LEN; r++) {
            for (int c = 0; c < D_MODEL; c++) {
                float orig = x.get(r, c);
                x.set(r, c, orig + eps);
                float fPlus = sumAll(attn.forward(x));
                x.set(r, c, orig - eps);
                float fMinus = sumAll(attn.forward(x));
                x.set(r, c, orig);
                assertEquals((fPlus - fMinus) / (2 * eps), dX.get(r, c), tol,
                        "MLA input grad mismatch at [" + r + "," + c + "]");
            }
        }
    }

    private static float sumAll(Tensor t) {
        float s = 0.0f;
        for (int r = 0; r < t.rows; r++)
            for (int c = 0; c < t.cols; c++)
                s += t.data[r * t.cols + c];
        return s;
    }
}

