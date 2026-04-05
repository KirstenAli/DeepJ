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

        cache = new ForwardCache(x, cQ, cKV, qhRope, khRope, attnProb, outH, merged);

        return merged.matmul(Wo.value);
    }

    // ── Backward ───────────────────────────────────────────────────

    @Override
    public Tensor backward(Tensor dOut) {
        int seqLen = cache.x.rows;

        // Output projection
        Wo.grad = Wo.grad.add(cache.merged.transpose().matmul(dOut));
        Tensor dMerged = dOut.matmul(Wo.value.transpose());

        // Attention backward
        Tensor dOutH = splitHeads(dMerged, seqLen);
        Tensor vh    = splitHeads(cache.cKV().matmul(Wuv.value), seqLen);

        Tensor dAttn = Tensor.zeros(nHeads * seqLen, seqLen);
        Tensor dVh   = Tensor.zeros(nHeads * seqLen, headDim);
        for (int h = 0; h < nHeads; h++) {
            Tensor doBlock = extractBlock(dOutH,          h * seqLen, seqLen, headDim);
            Tensor vBlock  = extractBlock(vh,             h * seqLen, seqLen, headDim);
            Tensor aBlock  = extractBlock(cache.attnProb, h * seqLen, seqLen, seqLen);
            insertBlock(dAttn, doBlock.matmul(vBlock.transpose()),  h * seqLen, seqLen, seqLen);
            insertBlock(dVh,   aBlock.transpose().matmul(doBlock),  h * seqLen, seqLen, headDim);
        }

        // Softmax + scale
        Tensor dScores = softmax.backward(dAttn).multiplyScalar(scale);

        // Q/K gradients from scores
        Tensor dQhRope = Tensor.zeros(nHeads * seqLen, headDim);
        Tensor dKhRope = Tensor.zeros(nHeads * seqLen, headDim);
        for (int h = 0; h < nHeads; h++) {
            Tensor dsBlock = extractBlock(dScores,       h * seqLen, seqLen, seqLen);
            Tensor qBlock  = extractBlock(cache.qhRope,  h * seqLen, seqLen, headDim);
            Tensor kBlock  = extractBlock(cache.khRope,  h * seqLen, seqLen, headDim);
            insertBlock(dQhRope, dsBlock.matmul(kBlock),             h * seqLen, seqLen, headDim);
            insertBlock(dKhRope, dsBlock.transpose().matmul(qBlock), h * seqLen, seqLen, headDim);
        }

        // RoPE backward
        Tensor dQh = rope.applyBackward(dQhRope, seqLen, nHeads);
        Tensor dKh = rope.applyBackward(dKhRope, seqLen, nHeads);

        // Merge heads
        Tensor dQ = mergeHeads(dQh, seqLen);
        Tensor dK = mergeHeads(dKh, seqLen);
        Tensor dV = mergeHeads(dVh, seqLen);

        // Q low-rank backward
        Wuq.grad  = Wuq.grad.add(cache.cQ.transpose().matmul(dQ));
        Tensor dcQ = dQ.matmul(Wuq.value.transpose());
        Wdq.grad  = Wdq.grad.add(cache.x.transpose().matmul(dcQ));
        Tensor dxQ = dcQ.matmul(Wdq.value.transpose());

        // KV shared compression backward
        Tensor cKV = cache.cKV;
        Wuk.grad  = Wuk.grad.add(cKV.transpose().matmul(dK));
        Wuv.grad  = Wuv.grad.add(cKV.transpose().matmul(dV));
        Tensor dcKV = dK.matmul(Wuk.value.transpose()).add(dV.matmul(Wuv.value.transpose()));
        Wdkv.grad = Wdkv.grad.add(cache.x.transpose().matmul(dcKV));
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
        Tensor out = Tensor.zeros(nHeads * seqLen, seqLen);
        for (int h = 0; h < nHeads; h++) {
            Tensor q = extractBlock(qh, h * seqLen, seqLen, headDim);
            Tensor k = extractBlock(kh, h * seqLen, seqLen, headDim);
            insertBlock(out, q.matmul(k.transpose()).multiplyScalar(scale), h * seqLen, seqLen, seqLen);
        }
        return out;
    }

    private Tensor applyMask(Tensor scores, int seqLen) {
        Tensor mask = Tensor.causalMask(seqLen);
        Tensor out  = Tensor.zeros(nHeads * seqLen, seqLen);
        for (int h = 0; h < nHeads; h++) {
            Tensor block = extractBlock(scores, h * seqLen, seqLen, seqLen);
            insertBlock(out, block.add(mask), h * seqLen, seqLen, seqLen);
        }
        return out;
    }

    private Tensor applyAttentionToValues(Tensor attnProb, Tensor vh, int seqLen) {
        Tensor out = Tensor.zeros(nHeads * seqLen, headDim);
        for (int h = 0; h < nHeads; h++) {
            Tensor a = extractBlock(attnProb, h * seqLen, seqLen, seqLen);
            Tensor v = extractBlock(vh,       h * seqLen, seqLen, headDim);
            insertBlock(out, a.matmul(v), h * seqLen, seqLen, headDim);
        }
        return out;
    }

    // ── Head reshape ───────────────────────────────────────────────

    private Tensor splitHeads(Tensor t, int seqLen) {
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

    private Tensor extractBlock(Tensor t, int rowStart, int numRows, int numCols) {
        Tensor block = Tensor.zeros(numRows, numCols);
        for (int r = 0; r < numRows; r++) {
            Tensor srcRow = t.getRow(rowStart + r);
            for (int c = 0; c < numCols; c++) {
                block.set(r, c, srcRow.get(0, c));
            }
        }
        return block;
    }

    private void insertBlock(Tensor target, Tensor block, int rowStart, int numRows, int numCols) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                target.set(rowStart + r, c, block.get(r, c));
            }
        }
    }

    // ── Cache ──────────────────────────────────────────────────────


    private record ForwardCache(
            Tensor x,
            Tensor cQ,
            Tensor cKV,
            Tensor qhRope,
            Tensor khRope,
            Tensor attnProb,
            Tensor outH,
            Tensor merged) {}
}

