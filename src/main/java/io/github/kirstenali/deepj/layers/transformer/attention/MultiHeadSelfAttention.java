package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.List;
import java.util.Random;

/**
 * Multi-head causal self-attention for a single sequence (no batch dimension).
 * Input/Output shape: [seqLen x dModel]
 */
public class MultiHeadSelfAttention implements Layer {

    private final int dModel;
    /** Exposed to subclasses that need head count for custom Q/K transforms (e.g. RoPE). */
    protected final int nHeads;
    /** Exposed to subclasses that need per-head dimension for custom Q/K transforms. */
    protected final int headDim;
    private final boolean causalMask;
    private final double scale;

    private final Parameter Wq;
    private final Parameter Wk;
    private final Parameter Wv;
    private final Parameter Wo;

    /** All tensors cached during forward that are needed for backward. */
    private ForwardCache cache;

    private final ActivationFunction softmax;

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd) {
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;
        this.scale = 1.0 / Math.sqrt(headDim);

        this.Wq = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wk = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wv = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wo = new Parameter(Tensor.random(dModel, dModel, rnd));

        this.softmax = new Softmax();
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);

        Projections proj = projectInputs(x);

        Tensor qh = splitHeads(proj.Q);
        Tensor kh = splitHeads(proj.K);
        Tensor vh = splitHeads(proj.V);

        // Allow subclasses to transform Q/K (no-op by default).
        Tensor cachedQh = transformQueryKey(qh, x.rows);
        Tensor cachedKh = transformQueryKey(kh, x.rows);

        Tensor scores = computeScaledDotProductScores(cachedQh, cachedKh, x.rows);
        if (causalMask) {
            scores = applyCausalMask(scores, x.rows);
        }

        Tensor attnProb      = softmax.forward(scores);
        Tensor outH          = applyAttentionToValues(attnProb, vh, x.rows);
        Tensor mergedBeforeWo = mergeHeads(outH, x.rows);

        cache = new ForwardCache(x, proj.Q, proj.K, proj.V,
                cachedQh, cachedKh, attnProb, outH, mergedBeforeWo);

        return mergedBeforeWo.matmul(Wo.value);
    }

    // -------------------------------------------------------------------------
    // Backward
    // -------------------------------------------------------------------------

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = cache.x.rows;

        Tensor dMerged = backwardOutputProjection(dOut);
        Tensor dOutH   = splitHeads(dMerged);
        Tensor vh      = splitHeads(cache.V);

        AttentionBackwardResult attnBackward = backwardAttentionAndValues(dOutH, vh, seqLen);

        // Use cached (post-hook) heads so score gradients stay in the transformed space.
        QueryKeyBackwardResult qkBackward = backwardQueriesAndKeys(
                attnBackward.dScores, cache.cachedQh, cache.cachedKh, seqLen);

        // Allow subclasses to invert their Q/K transform on the gradients (no-op by default).
        Tensor dQh = transformQueryKeyBackward(qkBackward.dQh, seqLen);
        Tensor dKh = transformQueryKeyBackward(qkBackward.dKh, seqLen);

        Tensor dQ = mergeHeads(dQh, seqLen);
        Tensor dK = mergeHeads(dKh, seqLen);
        Tensor dV = mergeHeads(attnBackward.dVh, seqLen);

        accumulateProjectionGrads(dQ, dK, dV);

        return dQ.matmul(Wq.value.transpose())
                .add(dK.matmul(Wk.value.transpose()))
                .add(dV.matmul(Wv.value.transpose()));
    }

    // -------------------------------------------------------------------------
    // Subclass hooks
    // -------------------------------------------------------------------------

    /**
     * Hook called on split-head Q and K tensors ({@code [nHeads·seqLen × headDim]})
     * before the scaled dot-product. No-op by default; override to apply RoPE etc.
     */
    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return heads;
    }

    /**
     * Inverse of {@link #transformQueryKey}, called on Q/K gradients in backward.
     * No-op by default; override to invert whatever forward transform was applied.
     */
    protected Tensor transformQueryKeyBackward(Tensor gradHeads, int seqLen) {
        return gradHeads;
    }

    // -------------------------------------------------------------------------
    // Forward helpers
    // -------------------------------------------------------------------------

    private void validateInput(Tensor x) {
        if (x.cols != dModel) {
            throw new IllegalArgumentException("Expected cols=" + dModel + " got " + x.cols);
        }
    }

    private Projections projectInputs(Tensor x) {
        return new Projections(
                x.matmul(Wq.value),
                x.matmul(Wk.value),
                x.matmul(Wv.value));
    }

    private Tensor computeScaledDotProductScores(Tensor qh, Tensor kh, int seqLen) {
        return mapHeadBlocks(nHeads, seqLen, seqLen, h -> {
            Tensor qBlock = extractBlock(qh, h * seqLen, seqLen, headDim);
            Tensor kBlock = extractBlock(kh, h * seqLen, seqLen, headDim);
            return qBlock.matmul(kBlock.transpose()).multiplyScalar(scale);
        });
    }

    private Tensor applyCausalMask(Tensor scores, int seqLen) {
        Tensor mask = Tensor.causalMask(seqLen);
        return mapHeadBlocks(nHeads, seqLen, seqLen, h -> {
            Tensor block = extractBlock(scores, h * seqLen, seqLen, seqLen);
            return block.add(mask);
        });
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        return mapHeadBlocks(nHeads, seqLen, headDim, h -> {
            Tensor aBlock = extractBlock(attnProb, h * seqLen, seqLen, seqLen);
            Tensor vBlock = extractBlock(vh,       h * seqLen, seqLen, headDim);
            return aBlock.matmul(vBlock);
        });
    }

    // -------------------------------------------------------------------------
    // Backward helpers
    // -------------------------------------------------------------------------

    private Tensor backwardOutputProjection(Tensor dOut) {
        Wo.grad = Wo.grad.add(cache.mergedBeforeWo.transpose().matmul(dOut));
        return dOut.matmul(Wo.value.transpose());
    }

    private AttentionBackwardResult backwardAttentionAndValues(Tensor dOutH, Tensor vh, int seqLen) {
        Tensor dAttn = Tensor.zeros(nHeads * seqLen, seqLen);
        Tensor dVh   = Tensor.zeros(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor doBlock = extractBlock(dOutH,          h * seqLen, seqLen, headDim);
            Tensor vBlock  = extractBlock(vh,             h * seqLen, seqLen, headDim);
            Tensor aBlock  = extractBlock(cache.attnProb, h * seqLen, seqLen, seqLen);

            insertBlock(dAttn, doBlock.matmul(vBlock.transpose()),  h * seqLen, seqLen, seqLen);
            insertBlock(dVh,   aBlock.transpose().matmul(doBlock),  h * seqLen, seqLen, headDim);
        }

        Tensor dScores = softmax.backward(dAttn).multiplyScalar(scale);
        return new AttentionBackwardResult(dScores, dVh);
    }

    private QueryKeyBackwardResult backwardQueriesAndKeys(
            Tensor dScores, Tensor qh, Tensor kh, int seqLen) {
        Tensor dQh = Tensor.zeros(nHeads * seqLen, headDim);
        Tensor dKh = Tensor.zeros(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor dsBlock = extractBlock(dScores, h * seqLen, seqLen, seqLen);
            Tensor qBlock  = extractBlock(qh,      h * seqLen, seqLen, headDim);
            Tensor kBlock  = extractBlock(kh,      h * seqLen, seqLen, headDim);

            insertBlock(dQh, dsBlock.matmul(kBlock),             h * seqLen, seqLen, headDim);
            insertBlock(dKh, dsBlock.transpose().matmul(qBlock), h * seqLen, seqLen, headDim);
        }

        return new QueryKeyBackwardResult(dQh, dKh);
    }

    private void accumulateProjectionGrads(Tensor dQ, Tensor dK, Tensor dV) {
        Tensor xT = cache.x.transpose();
        Wq.grad = Wq.grad.add(xT.matmul(dQ));
        Wk.grad = Wk.grad.add(xT.matmul(dK));
        Wv.grad = Wv.grad.add(xT.matmul(dV));
    }

    // -------------------------------------------------------------------------
    // Head reshape primitives
    // -------------------------------------------------------------------------

    /**
     * Apply {@code blockFn} to each head block and collect results into a single
     * {@code [nHeads*seqLen × outCols]} tensor.  Replaces the repeated
     * {@code for h … extractBlock … process … insertBlock} pattern.
     */
    private Tensor mapHeadBlocks(int nHeads, int seqLen, int outCols,
                                 HeadBlockFn blockFn) {
        Tensor out = Tensor.zeros(nHeads * seqLen, outCols);
        for (int h = 0; h < nHeads; h++) {
            insertBlock(out, blockFn.apply(h), h * seqLen, seqLen, outCols);
        }
        return out;
    }

    @FunctionalInterface
    private interface HeadBlockFn {
        Tensor apply(int headIndex);
    }

    /**
     * Split [seqLen x dModel] into [nHeads*seqLen x headDim] via row/col reindexing.
     */
    private Tensor splitHeads(Tensor t) {
        int seqLen = t.rows;
        Tensor out = Tensor.zeros(nHeads * seqLen, headDim);

        for (int i = 0; i < seqLen; i++) {
            Tensor srcRow = t.getRow(i);
            for (int h = 0; h < nHeads; h++) {
                int dstRow = h * seqLen + i;
                for (int d = 0; d < headDim; d++) {
                    out.set(dstRow, d, srcRow.get(0, h * headDim + d));
                }
            }
        }
        return out;
    }

    /**
     * Merge [nHeads*seqLen x headDim] back into [seqLen x dModel].
     */
    private Tensor mergeHeads(Tensor t, int seqLen) {
        Tensor out = Tensor.zeros(seqLen, dModel);

        for (int i = 0; i < seqLen; i++) {
            for (int h = 0; h < nHeads; h++) {
                int srcRow = h * seqLen + i;
                for (int d = 0; d < headDim; d++) {
                    out.set(i, h * headDim + d, t.get(srcRow, d));
                }
            }
        }
        return out;
    }

    /**
     * Extract a sub-block: rows [rowStart .. rowStart+numRows), cols [0..numCols).
     */
    private Tensor extractBlock(Tensor t, int rowStart, int numRows, int numCols) {
        Tensor block = Tensor.zeros(numRows, numCols);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                block.set(r, c, t.get(rowStart + r, c));
            }
        }
        return block;
    }

    /**
     * Insert {@code block} into {@code target} starting at {@code rowStart}.
     */
    private void insertBlock(Tensor target, Tensor block, int rowStart, int numRows, int numCols) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                target.set(rowStart + r, c, block.get(r, c));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------

    @Override
    public List<Parameter> parameters() {
        return List.of(Wq, Wk, Wv, Wo);
    }

    // -------------------------------------------------------------------------
    // Records
    // -------------------------------------------------------------------------

    /** Tensors cached during forward that are required by backward. */
    private record ForwardCache(
            Tensor x,
            Tensor Q, Tensor K, Tensor V,
            Tensor cachedQh, Tensor cachedKh,
            Tensor attnProb,
            Tensor outH,
            Tensor mergedBeforeWo) {
    }

    private record Projections(Tensor Q, Tensor K, Tensor V) {}

    private record AttentionBackwardResult(Tensor dScores, Tensor dVh) {}

    private record QueryKeyBackwardResult(Tensor dQh, Tensor dKh) {}
}

