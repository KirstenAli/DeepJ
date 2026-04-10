package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Shared flat-array head-reshape and attention-computation primitives used by every
 * multi-head attention variant.
 */
final class HeadOps {

    private HeadOps() {}

    private static final Map<Integer, Tensor> CAUSAL_MASK_CACHE = new ConcurrentHashMap<>();

    // ── Result carriers ────────────────────────────────────────────────────

    record AttentionGrads(Tensor dScores, Tensor dVh) {}
    record QKGrads(Tensor dQh, Tensor dKh) {}

    // ── Head reshape ───────────────────────────────────────────────────────

    /**
     * Split {@code [seqLen × dModel]} into {@code [nHeads·seqLen × headDim]}.
     * <pre>
     *   src:  t.data[i*dModel + h*headDim]          (token i, head h)
     *   dst: out.data[(h*seqLen + i)*headDim]
     * </pre>
     */
    static Tensor splitHeads(Tensor t, int seqLen, int nHeads, int headDim, int dModel) {
        // Required for lazy GPU backends: direct data[] access must read fresh CPU values.
        t.materialize();
        Tensor out = new Tensor(nHeads * seqLen, headDim);
        for (int i = 0; i < seqLen; i++) {
            int srcBase = i * dModel;
            for (int h = 0; h < nHeads; h++) {
                System.arraycopy(t.data, srcBase + h * headDim,
                                 out.data, (h * seqLen + i) * headDim,
                                 headDim);
            }
        }
        return out;
    }

    /**
     * Merge {@code [nHeads·seqLen × headDim]} back into {@code [seqLen × dModel]}.
     * Exact inverse of {@link #splitHeads}.
     */
    static Tensor mergeHeads(Tensor t, int seqLen, int nHeads, int headDim, int dModel) {
        t.materialize();
        Tensor out = new Tensor(seqLen, dModel);
        for (int i = 0; i < seqLen; i++) {
            int dstBase = i * dModel;
            for (int h = 0; h < nHeads; h++) {
                System.arraycopy(t.data, (h * seqLen + i) * headDim,
                                 out.data, dstBase + h * headDim,
                                 headDim);
            }
        }
        return out;
    }

    /**
     * Extract rows {@code [rowStart .. rowStart+numRows)} into a new
     * {@code [numRows × numCols]} tensor.
     * Contiguous in flat storage → single {@code arraycopy}.
     */
    static Tensor extractBlock(Tensor t, int rowStart, int numRows, int numCols) {
        t.materialize();
        Tensor block = new Tensor(numRows, numCols);
        System.arraycopy(t.data, rowStart * numCols, block.data, 0, numRows * numCols);
        return block;
    }

    /**
     * Insert a {@code [numRows × numCols]} block into {@code target} starting at {@code rowStart}.
     * Contiguous in flat storage → single {@code arraycopy}.
     */
    static void insertBlock(Tensor target, Tensor block, int rowStart, int numRows, int numCols) {
        block.materialize();
        System.arraycopy(block.data, 0, target.data, rowStart * numCols, numRows * numCols);
    }

    // ── Forward attention ops ──────────────────────────────────────────────

    /** Scaled dot-product scores: Q·Kᵀ / √headDim for all heads. */
    static Tensor scaledDotProductScores(Tensor qh, Tensor kh,
                                         int nHeads, int seqLen, int headDim, double scale) {
        Tensor out = new Tensor(nHeads * seqLen, seqLen);
        for (int h = 0; h < nHeads; h++) {
            Tensor q = extractBlock(qh, h * seqLen, seqLen, headDim);
            Tensor k = extractBlock(kh, h * seqLen, seqLen, headDim);
            Tensor s = q.matmul(k.transpose());
            s.multiplyScalarInPlace(scale);
            s.materialize();
            insertBlock(out, s, h * seqLen, seqLen, seqLen);
        }
        return out;
    }

    /** Add a shared causal mask to every head block. */
    static Tensor applyCausalMask(Tensor scores, int nHeads, int seqLen) {
        Tensor mask = CAUSAL_MASK_CACHE.computeIfAbsent(seqLen, HeadOps::buildCausalMask);
        Tensor out  = new Tensor(nHeads * seqLen, seqLen);
        for (int h = 0; h < nHeads; h++) {
            Tensor block = extractBlock(scores, h * seqLen, seqLen, seqLen);
            Tensor masked = block.add(mask);
            masked.materialize();
            insertBlock(out, masked, h * seqLen, seqLen, seqLen);
        }
        return out;
    }

    private static Tensor buildCausalMask(int seqLen) {
        Tensor mask = new Tensor(seqLen, seqLen);
        for (int r = 0; r < seqLen; r++) {
            int base = r * seqLen;
            for (int c = 0; c < seqLen; c++) {
                mask.data[base + c] = (c > r) ? -1e9f : 0.0f;
            }
        }
        return mask;
    }

    /** Weighted sum of values: attnProb · V for all heads. */
    static Tensor applyAttentionToValues(Tensor attnProb, Tensor vh,
                                         int nHeads, int seqLen, int headDim) {
        Tensor out = new Tensor(nHeads * seqLen, headDim);
        for (int h = 0; h < nHeads; h++) {
            Tensor a = extractBlock(attnProb, h * seqLen, seqLen, seqLen);
            Tensor v = extractBlock(vh,       h * seqLen, seqLen, headDim);
            Tensor y = a.matmul(v);
            y.materialize();
            insertBlock(out, y, h * seqLen, seqLen, headDim);
        }
        return out;
    }

    // ── Backward attention ops ─────────────────────────────────────────────

    /** Backprop through attn·V and the softmax+scale. */
    static AttentionGrads backwardAttentionAndValues(
            Tensor dOutH, Tensor vh, Tensor attnProb,
            ActivationFunction softmax, double scale,
            int nHeads, int seqLen, int headDim) {

        Tensor dAttn = new Tensor(nHeads * seqLen, seqLen);
        Tensor dVh   = new Tensor(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor doBlock = extractBlock(dOutH,    h * seqLen, seqLen, headDim);
            Tensor vBlock  = extractBlock(vh,       h * seqLen, seqLen, headDim);
            Tensor aBlock  = extractBlock(attnProb, h * seqLen, seqLen, seqLen);

            Tensor dAttnBlock = doBlock.matmul(vBlock.transpose());
            Tensor dVhBlock   = aBlock.transpose().matmul(doBlock);
            dAttnBlock.materialize();
            dVhBlock.materialize();
            insertBlock(dAttn, dAttnBlock, h * seqLen, seqLen, seqLen);
            insertBlock(dVh,   dVhBlock,   h * seqLen, seqLen, headDim);
        }

        Tensor dScores = softmax.backward(dAttn);
        dScores.multiplyScalarInPlace(scale);
        return new AttentionGrads(dScores, dVh);
    }

    /** Backprop through scaled dot-product scores: grad w.r.t. Q and K. */
    static QKGrads backwardQueriesAndKeys(
            Tensor dScores, Tensor qh, Tensor kh,
            int nHeads, int seqLen, int headDim) {

        Tensor dQh = new Tensor(nHeads * seqLen, headDim);
        Tensor dKh = new Tensor(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor ds = extractBlock(dScores, h * seqLen, seqLen, seqLen);
            Tensor q  = extractBlock(qh,      h * seqLen, seqLen, headDim);
            Tensor k  = extractBlock(kh,      h * seqLen, seqLen, headDim);

            Tensor dQhBlock = ds.matmul(k);
            Tensor dKhBlock = ds.transpose().matmul(q);
            dQhBlock.materialize();
            dKhBlock.materialize();
            insertBlock(dQh, dQhBlock, h * seqLen, seqLen, headDim);
            insertBlock(dKh, dKhBlock, h * seqLen, seqLen, headDim);
        }

        return new QKGrads(dQh, dKh);
    }
}
