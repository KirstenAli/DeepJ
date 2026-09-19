package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.List;
import java.util.Random;

public class MultiHeadSelfAttention implements Layer {

    private final int dModel;

    protected final int nHeads;

    protected final int headDim;
    private final boolean causalMask;
    private final float scale;

    private final Parameter Wq;
    private final Parameter Wk;
    private final Parameter Wv;
    private final Parameter Wo;

    private ForwardCache cache;

    private final ActivationFunction softmax;

    public MultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask, Random rnd) {
        validateDimensions(dModel, nHeads);

        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.causalMask = causalMask;
        this.scale = (float) (1.0 / Math.sqrt(headDim));

        this.Wq = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wk = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wv = new Parameter(Tensor.random(dModel, dModel, rnd));
        this.Wo = new Parameter(Tensor.random(dModel, dModel, rnd));

        this.softmax = new Softmax();
    }

    private static void validateDimensions(int dModel, int nHeads) {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (dModel % nHeads != 0) throw new IllegalArgumentException("dModel must be divisible by nHeads");
    }

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);

        Projections proj = projectInputs(x);

        Tensor qh = splitHeads(proj.Q);
        Tensor kh = splitHeads(proj.K);
        Tensor vh = splitHeads(proj.V);

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
                cachedQh, cachedKh, vh, attnProb, outH, mergedBeforeWo);

        return mergedBeforeWo.matmul(Wo.value);
    }

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = cache.x.rows;

        Tensor dMerged = backwardOutputProjection(dOut);
        Tensor dOutH   = splitHeads(dMerged);

        HeadOps.AttentionGrads attnBackward = backwardAttentionAndValues(dOutH, cache.vh, seqLen);

        HeadOps.QKGrads qkBackward = backwardQueriesAndKeys(
                attnBackward.dScores(), cache.cachedQh, cache.cachedKh, seqLen);

        Tensor dQh = transformQueryKeyBackward(qkBackward.dQh(), seqLen);
        Tensor dKh = transformQueryKeyBackward(qkBackward.dKh(), seqLen);

        Tensor dQ = mergeHeads(dQh, seqLen);
        Tensor dK = mergeHeads(dKh, seqLen);
        Tensor dV = mergeHeads(attnBackward.dVh(), seqLen);

        accumulateProjectionGrads(dQ, dK, dV);

        return dQ.matmul(Wq.value.transpose())
                .add(dK.matmul(Wk.value.transpose()))
                .add(dV.matmul(Wv.value.transpose()));
    }

    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return heads;
    }

    protected Tensor transformQueryKeyBackward(Tensor gradHeads, int seqLen) {
        return gradHeads;
    }

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
        return HeadOps.scaledDotProductScores(qh, kh, nHeads, seqLen, headDim, scale);
    }

    private Tensor applyCausalMask(Tensor scores, int seqLen) {
        return HeadOps.applyCausalMask(scores, nHeads, seqLen);
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        return HeadOps.applyAttentionToValues(attnProb, vh, nHeads, seqLen, headDim);
    }

    private Tensor backwardOutputProjection(Tensor dOut) {
        Wo.grad.addInPlace(cache.mergedBeforeWo.transpose().matmul(dOut));
        return dOut.matmul(Wo.value.transpose());
    }

    private HeadOps.AttentionGrads backwardAttentionAndValues(Tensor dOutH, Tensor vh, int seqLen) {
        return HeadOps.backwardAttentionAndValues(
                dOutH, vh, cache.attnProb, softmax, scale, nHeads, seqLen, headDim);
    }

    private HeadOps.QKGrads backwardQueriesAndKeys(Tensor dScores, Tensor qh, Tensor kh, int seqLen) {
        return HeadOps.backwardQueriesAndKeys(dScores, qh, kh, nHeads, seqLen, headDim);
    }

    private void accumulateProjectionGrads(Tensor dQ, Tensor dK, Tensor dV) {
        Tensor xT = cache.x.transpose();
        Wq.grad.addInPlace(xT.matmul(dQ));
        Wk.grad.addInPlace(xT.matmul(dK));
        Wv.grad.addInPlace(xT.matmul(dV));
    }

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

    private Tensor splitHeads(Tensor t) {
        return HeadOps.splitHeads(t, t.rows, nHeads, headDim, dModel);
    }

    private Tensor mergeHeads(Tensor t, int seqLen) {
        return HeadOps.mergeHeads(t, seqLen, nHeads, headDim, dModel);
    }

    private Tensor extractBlock(Tensor t, int rowStart, int numRows, int numCols) {
        return HeadOps.extractBlock(t, rowStart, numRows, numCols);
    }

    private void insertBlock(Tensor target, Tensor block, int rowStart, int numRows, int numCols) {
        HeadOps.insertBlock(target, block, rowStart, numRows, numCols);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(Wq, Wk, Wv, Wo);
    }

    private record ForwardCache(
            Tensor x,
            Tensor Q, Tensor K, Tensor V,
            Tensor cachedQh, Tensor cachedKh,
            Tensor vh,
            Tensor attnProb,
            Tensor outH,
            Tensor mergedBeforeWo) {}

    private record Projections(Tensor Q, Tensor K, Tensor V) {}
}
