package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.List;
import java.util.Random;

public final class MultiHeadLatentAttention implements Layer {

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final float scale;

    private final Parameter Wdq;
    private final Parameter Wuq;

    private final Parameter Wdkv;
    private final Parameter Wuk;
    private final Parameter Wuv;

    private final Parameter Wo;

    private final RotaryEmbedding rope;
    private final Softmax softmax;

    private ForwardCache cache;

    public MultiHeadLatentAttention(int dModel, int nHeads, int qRank, int kvRank,
                                    RotaryEmbedding rope, Random rnd) {
        validateDimensions(dModel, nHeads, qRank, kvRank, rope);

        this.dModel  = dModel;
        this.nHeads  = nHeads;
        this.headDim = dModel / nHeads;
        this.scale   = (float) (1.0 / Math.sqrt(headDim));
        this.rope    = rope;
        this.softmax = new Softmax();

        this.Wdq  = new Parameter(Tensor.random(dModel, qRank,  rnd));
        this.Wuq  = new Parameter(Tensor.random(qRank,  dModel, rnd));
        this.Wdkv = new Parameter(Tensor.random(dModel, kvRank, rnd));
        this.Wuk  = new Parameter(Tensor.random(kvRank, dModel, rnd));
        this.Wuv  = new Parameter(Tensor.random(kvRank, dModel, rnd));
        this.Wo   = new Parameter(Tensor.random(dModel, dModel, rnd));
    }

    private static void validateDimensions(int dModel, int nHeads, int qRank, int kvRank,
                                           RotaryEmbedding rope) {
        requirePositive(dModel, "dModel");
        requirePositive(nHeads, "nHeads");
        requirePositive(qRank, "qRank");
        requirePositive(kvRank, "kvRank");
        requireDivisible(dModel, nHeads);
        if (rope == null) throw new IllegalArgumentException("rope must not be null");
        if (rope.headDim() != dModel / nHeads) throw new IllegalArgumentException("RoPE head dimension mismatch");
    }

    private static void requirePositive(int value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be > 0");
    }

    private static void requireDivisible(int dModel, int nHeads) {
        if (dModel % nHeads != 0) throw new IllegalArgumentException("dModel must be divisible by nHeads");
    }

    @Override
    public Tensor forward(Tensor x) {
        int seqLen = x.rows;
        cache = projectAndAttend(x, seqLen);
        return cache.merged().matmul(Wo.value);
    }

    private ForwardCache projectAndAttend(Tensor x, int seqLen) {
        Tensor cQ = x.matmul(Wdq.value);
        Tensor Q  = cQ.matmul(Wuq.value);
        Tensor cKV = x.matmul(Wdkv.value);
        Tensor K   = cKV.matmul(Wuk.value);
        Tensor V   = cKV.matmul(Wuv.value);
        Tensor qh = splitHeads(Q, seqLen);
        Tensor kh = splitHeads(K, seqLen);
        Tensor vh = splitHeads(V, seqLen);
        return attend(x, cQ, cKV, qh, kh, vh, seqLen);
    }

    private ForwardCache attend(Tensor x, Tensor cQ, Tensor cKV, Tensor qh,
                                Tensor kh, Tensor vh, int seqLen) {
        Tensor qhRope = rope.apply(qh, seqLen, nHeads);
        Tensor khRope = rope.apply(kh, seqLen, nHeads);
        Tensor scores   = computeScores(qhRope, khRope);
        Tensor attnProb = softmax.forwardCausal(scores, seqLen, scale);
        Tensor outH     = applyAttentionToValues(attnProb, vh, seqLen);
        Tensor merged   = mergeHeads(outH, seqLen);
        return new ForwardCache(x, cQ, cKV, qhRope, khRope, vh, attnProb, outH, merged);
    }

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = cache.x.rows;
        Tensor dMerged = backwardOutputProjection(dOut);
        ProjectionGrads grads = backwardAttention(dMerged, seqLen);
        Tensor dxQ = backwardQueryPath(grads.dQ());
        Tensor dxKV = backwardKeyValuePath(grads.dK(), grads.dV());
        return dxQ.add(dxKV);
    }

    private Tensor backwardOutputProjection(Tensor dOut) {
        Wo.grad.addInPlace(cache.merged.transpose().matmul(dOut));
        return dOut.matmul(Wo.value.transpose());
    }

    private ProjectionGrads backwardAttention(Tensor dMerged, int seqLen) {
        Tensor dOutH = splitHeads(dMerged, seqLen);
        HeadOps.AttentionGrads attnGrads = HeadOps.backwardAttentionAndValues(
                dOutH, cache.vh, cache.attnProb, softmax, scale, nHeads);
        HeadOps.QKGrads qkGrads = HeadOps.backwardQueriesAndKeys(
                attnGrads.dScores(), cache.qhRope, cache.khRope, nHeads);
        Tensor dQh = rope.applyBackward(qkGrads.dQh(), seqLen, nHeads);
        Tensor dKh = rope.applyBackward(qkGrads.dKh(), seqLen, nHeads);
        return new ProjectionGrads(mergeHeads(dQh, seqLen), mergeHeads(dKh, seqLen),
                mergeHeads(attnGrads.dVh(), seqLen));
    }

    private Tensor backwardQueryPath(Tensor dQ) {
        Wuq.grad.addInPlace(cache.cQ.transpose().matmul(dQ));
        Tensor dcQ = dQ.matmul(Wuq.value.transpose());
        Wdq.grad.addInPlace(cache.x.transpose().matmul(dcQ));
        return dcQ.matmul(Wdq.value.transpose());
    }

    private Tensor backwardKeyValuePath(Tensor dK, Tensor dV) {
        Tensor cKV = cache.cKV;
        Wuk.grad.addInPlace(cKV.transpose().matmul(dK));
        Wuv.grad.addInPlace(cKV.transpose().matmul(dV));
        Tensor dcKV = dK.matmul(Wuk.value.transpose());
        dcKV.addInPlace(dV.matmul(Wuv.value.transpose()));
        Wdkv.grad.addInPlace(cache.x.transpose().matmul(dcKV));
        return dcKV.matmul(Wdkv.value.transpose());
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(Wdq, Wuq, Wdkv, Wuk, Wuv, Wo);
    }

    private Tensor computeScores(Tensor qh, Tensor kh) {
        return HeadOps.dotProductScores(qh, kh, nHeads);
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        return HeadOps.applyAttentionToValues(attnProb, vh, nHeads);
    }

    private Tensor splitHeads(Tensor t, int seqLen) {
        return HeadOps.splitHeads(t, nHeads);
    }

    private Tensor mergeHeads(Tensor t, int seqLen) {
        return HeadOps.mergeHeads(t, nHeads);
    }

    private record ForwardCache(
            Tensor x,
            Tensor cQ,
            Tensor cKV,
            Tensor qhRope,
            Tensor khRope,
            Tensor vh,
            Tensor attnProb,
            Tensor outH,
            Tensor merged) {}

    private record ProjectionGrads(Tensor dQ, Tensor dK, Tensor dV) {}
}
