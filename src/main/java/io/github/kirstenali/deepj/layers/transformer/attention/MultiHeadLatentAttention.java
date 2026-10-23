package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.List;
import java.util.Random;

public final class MultiHeadLatentAttention implements Layer {

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

    public MultiHeadLatentAttention(int dModel, int nHeads, int qRank, int kvRank,
                                    RotaryEmbedding rope, Random rnd) {
        validateDimensions(dModel, nHeads, qRank, kvRank, rope);

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
    public Tensor forward(Tensor input) {
        Tensor query = input.matmul(Wdq.value).matmul(Wuq.value);
        Tensor compressed = input.matmul(Wdkv.value);
        Tensor key = compressed.matmul(Wuk.value);
        Tensor value = compressed.matmul(Wuv.value);
        Tensor context = attend(query, key, value, input.rows);
        return HeadOps.mergeHeads(context, nHeads).matmul(Wo.value);
    }

    private Tensor attend(Tensor query, Tensor key, Tensor value, int length) {
        Tensor queries = rope.apply(HeadOps.splitHeads(query, nHeads), length, nHeads);
        Tensor keys = rope.apply(HeadOps.splitHeads(key, nHeads), length, nHeads);
        Tensor scores = HeadOps.dotProductScores(queries, keys, nHeads);
        Tensor probabilities = softmax.forwardCausal(scores, length, scale);
        return HeadOps.applyAttentionToValues(probabilities, HeadOps.splitHeads(value, nHeads), nHeads);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(Wdq, Wuq, Wdkv, Wuk, Wuv, Wo);
    }
}
