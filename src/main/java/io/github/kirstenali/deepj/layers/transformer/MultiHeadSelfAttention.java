package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multi-head causal self-attention for a single sequence (no batch dimension).
 * Input/Output shape: [seqLen x dModel]

 * Notes:
 *  - This implementation is intentionally explicit (loops) to keep the code dependency-free.
 *  - For real throughput, replace inner loops with vectorized / native kernels.
 */
public final class MultiHeadSelfAttention implements Layer {

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final boolean causalMask;

    private final Parameter Wq;
    private final Parameter Wk;
    private final Parameter Wv;
    private final Parameter Wo;

    private Tensor x;
    private Tensor Q, K, V;
    private Tensor attnProb;       // [heads*seqLen x seqLen]
    private Tensor outH;           // [heads*seqLen x headDim]
    private Tensor mergedBeforeWo; // [seqLen x dModel]

    private final ActivationFunction softmax;

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd) {
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;

        this.Wq = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wk = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wv = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wo = new Parameter(Tensor.random(dModel, dModel, rnd));

        this.softmax = new Softmax();
    }

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);
        this.x = x;

        projectInputs(x);

        Tensor qh = splitHeads(Q);
        Tensor kh = splitHeads(K);
        Tensor vh = splitHeads(V);

        Tensor scores = computeScaledDotProductScores(qh, kh, x.rows);

        if (causalMask) {
            scores = applyCausalMask(scores, x.rows);
        }

        attnProb = softmax.forward(scores);
        outH = applyAttentionToValues(attnProb, vh, x.rows);

        mergedBeforeWo = mergeHeads(outH, x.rows);
        return mergedBeforeWo.matmul(Wo.value);
    }

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = x.rows;

        Tensor dMerged = backwardOutputProjection(dOut);
        Tensor dOutH = splitHeads(dMerged);
        Tensor vh = splitHeads(V);

        AttentionBackwardResult attnBackward = backwardAttentionAndValues(dOutH, vh, seqLen);

        Tensor qh = splitHeads(Q);
        Tensor kh = splitHeads(K);

        QueryKeyBackwardResult qkBackward = backwardQueriesAndKeys(
                attnBackward.dScores, qh, kh, seqLen
        );

        Tensor dQ = mergeHeads(qkBackward.dQh, seqLen);
        Tensor dK = mergeHeads(qkBackward.dKh, seqLen);
        Tensor dV = mergeHeads(attnBackward.dVh, seqLen);

        accumulateProjectionGrads(dQ, dK, dV);

        return dQ.matmul(Wq.value.transpose())
                .add(dK.matmul(Wk.value.transpose()))
                .add(dV.matmul(Wv.value.transpose()));
    }

    private void validateInput(Tensor x) {
        if (x.cols != dModel) {
            throw new IllegalArgumentException("Expected cols=" + dModel + " got " + x.cols);
        }
    }

    private void projectInputs(Tensor x) {
        Q = x.matmul(Wq.value);
        K = x.matmul(Wk.value);
        V = x.matmul(Wv.value);
    }

    private Tensor computeScaledDotProductScores(Tensor qh, Tensor kh, int seqLen) {
        // qh, kh: [nHeads*seqLen x headDim]
        // For each head block: scores_block = qh_block * kh_block^T, then scale
        // We compute this as a block-diagonal matmul by iterating heads
        Tensor scores = Tensor.zeros(nHeads * seqLen, seqLen);
        double scale = 1.0 / Math.sqrt(headDim);

        for (int h = 0; h < nHeads; h++) {
            // Extract head blocks as sub-tensors
            Tensor qBlock = extractBlock(qh, h * seqLen, seqLen, headDim);
            Tensor kBlock = extractBlock(kh, h * seqLen, seqLen, headDim);

            // scores_block = qBlock * kBlock^T * scale
            Tensor block = qBlock.matmul(kBlock.transpose()).multiplyScalar(scale);

            // Copy block into scores
            insertBlock(scores, block, h * seqLen, seqLen, seqLen);
        }

        return scores;
    }

    private Tensor applyCausalMask(Tensor scores, int seqLen) {
        Tensor mask = Tensor.causalMask(seqLen);

        // Apply mask to each head block
        Tensor result = Tensor.zeros(scores.rows, scores.cols);
        for (int h = 0; h < nHeads; h++) {
            Tensor block = extractBlock(scores, h * seqLen, seqLen, seqLen);
            Tensor masked = block.add(mask);
            insertBlock(result, masked, h * seqLen, seqLen, seqLen);
        }
        return result;
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        // attnProb: [nHeads*seqLen x seqLen], vh: [nHeads*seqLen x headDim]
        Tensor outH = Tensor.zeros(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor aBlock = extractBlock(attnProb, h * seqLen, seqLen, seqLen);
            Tensor vBlock = extractBlock(vh, h * seqLen, seqLen, headDim);

            Tensor block = aBlock.matmul(vBlock);
            insertBlock(outH, block, h * seqLen, seqLen, headDim);
        }

        return outH;
    }

    private Tensor backwardOutputProjection(Tensor dOut) {
        Wo.grad = Wo.grad.add(mergedBeforeWo.transpose().matmul(dOut));
        return dOut.matmul(Wo.value.transpose());
    }

    private AttentionBackwardResult backwardAttentionAndValues(Tensor dOutH, Tensor vh, int seqLen) {
        Tensor dAttn = Tensor.zeros(nHeads * seqLen, seqLen);
        Tensor dVh = Tensor.zeros(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor doBlock = extractBlock(dOutH, h * seqLen, seqLen, headDim);
            Tensor vBlock = extractBlock(vh, h * seqLen, seqLen, headDim);
            Tensor aBlock = extractBlock(attnProb, h * seqLen, seqLen, seqLen);

            // dAttn_block = doBlock * vBlock^T
            Tensor dAttnBlock = doBlock.matmul(vBlock.transpose());
            insertBlock(dAttn, dAttnBlock, h * seqLen, seqLen, seqLen);

            // dVh_block = aBlock^T * doBlock
            Tensor dVhBlock = aBlock.transpose().matmul(doBlock);
            insertBlock(dVh, dVhBlock, h * seqLen, seqLen, headDim);
        }

        Tensor dScores = softmax.backward(dAttn)
                .multiplyScalar(1.0 / Math.sqrt(headDim));

        return new AttentionBackwardResult(dScores, dVh);
    }

    private QueryKeyBackwardResult backwardQueriesAndKeys(
            Tensor dScores, Tensor qh, Tensor kh, int seqLen
    ) {
        Tensor dQh = Tensor.zeros(nHeads * seqLen, headDim);
        Tensor dKh = Tensor.zeros(nHeads * seqLen, headDim);

        for (int h = 0; h < nHeads; h++) {
            Tensor dsBlock = extractBlock(dScores, h * seqLen, seqLen, seqLen);
            Tensor qBlock = extractBlock(qh, h * seqLen, seqLen, headDim);
            Tensor kBlock = extractBlock(kh, h * seqLen, seqLen, headDim);

            // dQh_block = dsBlock * kBlock
            Tensor dQhBlock = dsBlock.matmul(kBlock);
            insertBlock(dQh, dQhBlock, h * seqLen, seqLen, headDim);

            // dKh_block = dsBlock^T * qBlock
            Tensor dKhBlock = dsBlock.transpose().matmul(qBlock);
            insertBlock(dKh, dKhBlock, h * seqLen, seqLen, headDim);
        }

        return new QueryKeyBackwardResult(dQh, dKh);
    }

    private void accumulateProjectionGrads(Tensor dQ, Tensor dK, Tensor dV) {
        Wq.grad = Wq.grad.add(x.transpose().matmul(dQ));
        Wk.grad = Wk.grad.add(x.transpose().matmul(dK));
        Wv.grad = Wv.grad.add(x.transpose().matmul(dV));
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
                // Copy cols [h*headDim .. (h+1)*headDim) from srcRow into out row dstRow
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
     * Extract a sub-block: rows [rowStart .. rowStart+numRows), all cols [0..numCols).
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
     * Insert a block into target at row offset.
     */
    private void insertBlock(Tensor target, Tensor block, int rowStart, int numRows, int numCols) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                target.set(rowStart + r, c, block.get(r, c));
            }
        }
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.add(Wq);
        ps.add(Wk);
        ps.add(Wv);
        ps.add(Wo);
        return ps;
    }

    private record AttentionBackwardResult(Tensor dScores, Tensor dVh) {
    }

    private record QueryKeyBackwardResult(Tensor dQh, Tensor dKh) {
    }
}