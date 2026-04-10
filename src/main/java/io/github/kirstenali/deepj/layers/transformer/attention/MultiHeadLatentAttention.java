package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.List;
import java.util.Random;

/**
 * Multi-Head Latent Attention (MLA) — the attention mechanism introduced in DeepSeek-V2/V3/R1.
 *
 * <p>Unlike standard MHA which projects Q, K, V directly from {@code x}, MLA first compresses
 * through a low-rank bottleneck then expands:
 * <pre>
 *   cQ  = x   · Wdq      [seqLen × qRank]   — Q  compression
 *   Q   = cQ  · Wuq      [seqLen × dModel]  — Q  expansion
 *
 *   cKV = x   · Wdkv     [seqLen × kvRank]  — shared KV compression
 *   K   = cKV · Wuk      [seqLen × dModel]  — K  expansion
 *   V   = cKV · Wuv      [seqLen × dModel]  — V  expansion
 * </pre>
 *
 * <p>RoPE is applied to Q and K after expansion. Scaled dot-product attention and the
 * output projection {@code Wo} are then identical to standard MHA.
 *
 * <p>The key inference benefit: only {@code cKV} (shape {@code seqLen × kvRank}) needs to be
 * cached per layer — much smaller than the full {@code K} and {@code V} tensors.
 *
 * <p><b>Parameters:</b> Wdq, Wuq, Wdkv, Wuk, Wuv, Wo (6 total, vs 4 in standard MHA).
 */
public final class MultiHeadLatentAttention implements Layer {

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final double scale;

    // Q low-rank projections
    private final Parameter Wdq;    // dModel → qRank
    private final Parameter Wuq;    // qRank  → dModel

    // KV shared low-rank projections
    private final Parameter Wdkv;   // dModel → kvRank
    private final Parameter Wuk;    // kvRank → dModel
    private final Parameter Wuv;    // kvRank → dModel

    // Output projection
    private final Parameter Wo;     // dModel → dModel

    private final RotaryEmbedding rope;
    private final Softmax softmax;

    private ForwardCache cache;

    /**
     * @param dModel   model dimension
     * @param nHeads   number of attention heads; must divide {@code dModel}
     * @param qRank    Q latent dimension (e.g. 1536 in DeepSeek-V2; use dModel/2 for small models)
     * @param kvRank   KV latent dimension (e.g. 512 in DeepSeek-V2; use dModel/4 for small models)
     * @param rope     pre-built rotary embedding sized for {@code dModel / nHeads}
     * @param rnd      random source for weight initialisation
     */
    public MultiHeadLatentAttention(int dModel, int nHeads, int qRank, int kvRank,
                                    RotaryEmbedding rope, Random rnd) {
        if (dModel % nHeads != 0)
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        if (qRank <= 0)  throw new IllegalArgumentException("qRank must be > 0");
        if (kvRank <= 0) throw new IllegalArgumentException("kvRank must be > 0");
        if (rope == null) throw new IllegalArgumentException("rope must not be null");

        this.dModel  = dModel;
        this.nHeads  = nHeads;
        this.headDim = dModel / nHeads;
        this.scale   = 1.0 / Math.sqrt(headDim);
        this.rope    = rope;
        this.softmax = new Softmax();

        this.Wdq  = new Parameter(Tensor.random(dModel, qRank,  rnd));
        this.Wuq  = new Parameter(Tensor.random(qRank,  dModel, rnd));
        this.Wdkv = new Parameter(Tensor.random(dModel, kvRank, rnd));
        this.Wuk  = new Parameter(Tensor.random(kvRank, dModel, rnd));
        this.Wuv  = new Parameter(Tensor.random(kvRank, dModel, rnd));
        this.Wo   = new Parameter(Tensor.random(dModel, dModel, rnd));
    }

    // ── Forward ────────────────────────────────────────────────────

    @Override
    public Tensor forward(Tensor x) {
        int seqLen = x.rows;

        // Q low-rank path
        Tensor cQ = x.matmul(Wdq.value);       // [seqLen × qRank]
        Tensor Q  = cQ.matmul(Wuq.value);      // [seqLen × dModel]

        // KV shared compression
        Tensor cKV = x.matmul(Wdkv.value);     // [seqLen × kvRank]
        Tensor K   = cKV.matmul(Wuk.value);    // [seqLen × dModel]
        Tensor V   = cKV.matmul(Wuv.value);    // [seqLen × dModel]

        // Split heads
        Tensor qh = splitHeads(Q, seqLen);
        Tensor kh = splitHeads(K, seqLen);
        Tensor vh = splitHeads(V, seqLen);

        // Apply RoPE to Q and K
        Tensor qhRope = rope.apply(qh, seqLen, nHeads);
        Tensor khRope = rope.apply(kh, seqLen, nHeads);

        // Scaled dot-product attention
        Tensor scores   = computeScores(qhRope, khRope, seqLen);
        Tensor masked   = applyMask(scores, seqLen);
        Tensor attnProb = softmax.forward(masked);
        Tensor outH     = applyAttentionToValues(attnProb, vh, seqLen);
        Tensor merged   = mergeHeads(outH, seqLen);

        cache = new ForwardCache(x, cQ, cKV, qhRope, khRope, vh, attnProb, outH, merged);

        return merged.matmul(Wo.value);
    }

    // ── Backward ───────────────────────────────────────────────────

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = cache.x.rows;

        // Output projection
        Wo.grad.addInPlace(cache.merged.transpose().matmul(dOut));
        Tensor dMerged = dOut.matmul(Wo.value.transpose());

        // Attention backward
        Tensor dOutH = splitHeads(dMerged, seqLen);

        HeadOps.AttentionGrads attnGrads = HeadOps.backwardAttentionAndValues(
                dOutH, cache.vh, cache.attnProb, softmax, scale, nHeads, seqLen, headDim);

        HeadOps.QKGrads qkGrads = HeadOps.backwardQueriesAndKeys(
                attnGrads.dScores(), cache.qhRope, cache.khRope, nHeads, seqLen, headDim);

        // RoPE backward
        Tensor dQh = rope.applyBackward(qkGrads.dQh(), seqLen, nHeads);
        Tensor dKh = rope.applyBackward(qkGrads.dKh(), seqLen, nHeads);

        // Merge heads
        Tensor dQ = mergeHeads(dQh, seqLen);
        Tensor dK = mergeHeads(dKh, seqLen);
        Tensor dV = mergeHeads(attnGrads.dVh(), seqLen);

        // Q low-rank backward
        Wuq.grad.addInPlace(cache.cQ.transpose().matmul(dQ));
        Tensor dcQ = dQ.matmul(Wuq.value.transpose());
        Wdq.grad.addInPlace(cache.x.transpose().matmul(dcQ));
        Tensor dxQ = dcQ.matmul(Wdq.value.transpose());

        // KV shared compression backward
        Tensor cKV = cache.cKV;
        Wuk.grad.addInPlace(cKV.transpose().matmul(dK));
        Wuv.grad.addInPlace(cKV.transpose().matmul(dV));
        Tensor dcKV = dK.matmul(Wuk.value.transpose());
        dcKV.addInPlace(dV.matmul(Wuv.value.transpose()));
        Wdkv.grad.addInPlace(cache.x.transpose().matmul(dcKV));
        Tensor dxKV = dcKV.matmul(Wdkv.value.transpose());

        return dxQ.add(dxKV);
    }

    // ── Parameters ─────────────────────────────────────────────────

    @Override
    public List<Parameter> parameters() {
        return List.of(Wdq, Wuq, Wdkv, Wuk, Wuv, Wo);
    }

    // ── Attention helpers ───────────────────────────────────────────

    private Tensor computeScores(Tensor qh, Tensor kh, int seqLen) {
        return HeadOps.scaledDotProductScores(qh, kh, nHeads, seqLen, headDim, scale);
    }

    private Tensor applyMask(Tensor scores, int seqLen) {
        return HeadOps.applyCausalMask(scores, nHeads, seqLen);
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        return HeadOps.applyAttentionToValues(attnProb, vh, nHeads, seqLen, headDim);
    }

    // ── Head reshape ───────────────────────────────────────────────

    private Tensor splitHeads(Tensor t, int seqLen) {
        return HeadOps.splitHeads(t, seqLen, nHeads, headDim, dModel);
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

    // ── Cache ──────────────────────────────────────────────────────


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
}

