package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.Random;

public final class RoPEMultiHeadSelfAttention extends MultiHeadSelfAttention {

    private final RotaryEmbedding rope;

    public RoPEMultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask,
                                      RotaryEmbedding rope, Random rnd) {
        super(dModel, nHeads, causalMask, rnd);
        if (rope == null) throw new IllegalArgumentException("rope must not be null");
        if (rope.headDim() != headDim) {
            throw new IllegalArgumentException("RoPE head dimension does not match attention");
        }
        this.rope = rope;
    }

    @Override
    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return rope.apply(heads, seqLen, nHeads);
    }

    @Override
    protected Tensor transformQueryKeyBackward(Tensor gradHeads, int seqLen) {
        return rope.applyBackward(gradHeads, seqLen, nHeads);
    }
}
