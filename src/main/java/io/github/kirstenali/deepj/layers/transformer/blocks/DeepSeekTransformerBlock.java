package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadLatentAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.Random;

public final class DeepSeekTransformerBlock extends AbstractTransformerBlock {

    private final RMSNorm1D             ln1;
    private final RMSNorm1D             ln2;
    private final MultiHeadLatentAttention attn;
    private final SwiGLULayer           mlp;

    public DeepSeekTransformerBlock(int dModel, int nHeads, int qRank, int kvRank,
                                    int dFF, int maxSeqLen, Random rnd) {
        RotaryEmbedding rope = createRope(dModel, nHeads, maxSeqLen);
        this.ln1  = new RMSNorm1D(dModel);
        this.ln2  = new RMSNorm1D(dModel);
        this.attn = new MultiHeadLatentAttention(dModel, nHeads, qRank, kvRank, rope, rnd);
        this.mlp  = new SwiGLULayer(dModel, dFF, rnd);
    }

    private static RotaryEmbedding createRope(int dModel, int nHeads, int maxSeqLen) {
        if (dModel <= 0 || nHeads <= 0 || dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be positive and divisible by nHeads");
        }
        return new RotaryEmbedding(dModel / nHeads, maxSeqLen);
    }

    @Override
    protected Layer[] subLayers() {
        return new Layer[]{ ln1, ln2, attn, mlp };
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor x2 = x.add(attn.forward(ln1.forward(x)));
        return x2.add(mlp.forward(ln2.forward(x2)));
    }

    @Override
    public Tensor backward(Tensor gradOut) {
        Tensor gMlp = mlp.backward(gradOut);
        Tensor gX2  = gradOut.add(ln2.backward(gMlp));

        Tensor gAttn = attn.backward(gX2);
        return gX2.add(ln1.backward(gAttn));
    }
}
