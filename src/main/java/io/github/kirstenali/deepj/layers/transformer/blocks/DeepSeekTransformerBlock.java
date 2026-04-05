package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadLatentAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.Random;

/**
 * Pre-LN Transformer block wired for DeepSeek-V2/V3/R1 style:
 * <pre>
 *   x = x + MLA( RMSNorm(x) )
 *   x = x + SwiGLU( RMSNorm(x) )
 * </pre>
 *
 * <p>Identical to {@link LlamaTransformerBlock} except attention uses
 * {@link MultiHeadLatentAttention} instead of RoPE-MHA, giving a smaller
 * KV cache footprint during inference.
 */
public final class DeepSeekTransformerBlock extends AbstractTransformerBlock {

    private final RMSNorm1D             ln1;
    private final RMSNorm1D             ln2;
    private final MultiHeadLatentAttention attn;
    private final SwiGLULayer           mlp;

    /**
     * @param dModel     model dimension
     * @param nHeads     attention heads (must divide dModel)
     * @param qRank      Q latent dimension
     * @param kvRank     KV latent dimension
     * @param dFF        SwiGLU intermediate dimension
     * @param maxSeqLen  maximum sequence length for the RoPE table
     * @param rnd        random source for weight initialisation
     */
    public DeepSeekTransformerBlock(int dModel, int nHeads, int qRank, int kvRank,
                                    int dFF, int maxSeqLen, Random rnd) {
        RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, maxSeqLen);
        this.ln1  = new RMSNorm1D(dModel);
        this.ln2  = new RMSNorm1D(dModel);
        this.attn = new MultiHeadLatentAttention(dModel, nHeads, qRank, kvRank, rope, rnd);
        this.mlp  = new SwiGLULayer(dModel, dFF, rnd);
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
