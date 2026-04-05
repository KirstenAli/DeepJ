package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Pre-LN Transformer block wired for Llama / Mistral / Qwen / DeepSeek style:
 * <pre>
 *   x = x + RoPE-Attn( RMSNorm(x) )
 *   x = x + SwiGLU(    RMSNorm(x) )
 * </pre>
 *
 * <p>Composes {@link RMSNorm1D}, {@link RoPEMultiHeadSelfAttention}, and
 * {@link SwiGLULayer} without touching the existing {@link TransformerBlock}.
 */
public final class LlamaTransformerBlock implements Layer {

    private final RMSNorm1D ln1;
    private final RMSNorm1D ln2;
    private final RoPEMultiHeadSelfAttention attn;
    private final SwiGLULayer mlp;

    /**
     * @param dModel     model dimension
     * @param nHeads     attention heads (must divide dModel)
     * @param dFF        SwiGLU intermediate dimension (typically ≈ 8/3 × dModel)
     * @param maxSeqLen  maximum sequence length for the RoPE table
     * @param rnd        random source for weight initialisation
     */
    public LlamaTransformerBlock(int dModel, int nHeads, int dFF, int maxSeqLen, Random rnd) {
        RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, maxSeqLen);
        this.ln1  = new RMSNorm1D(dModel);
        this.ln2  = new RMSNorm1D(dModel);
        this.attn = new RoPEMultiHeadSelfAttention(dModel, nHeads, true, rope, rnd);
        this.mlp  = new SwiGLULayer(dModel, dFF, rnd);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor x2 = x.add(attn.forward(ln1.forward(x)));
        return x2.add(mlp.forward(ln2.forward(x2)));
    }

    @Override
    public Tensor backward(Tensor gradOut) {
        // residual add2: y = x2 + mlp(ln2(x2))
        Tensor gMlp = mlp.backward(gradOut);
        Tensor gX2  = gradOut.add(ln2.backward(gMlp));

        // residual add1: x2 = x + attn(ln1(x))
        Tensor gAttn = attn.backward(gX2);
        return gX2.add(ln1.backward(gAttn));
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(ln1.parameters());
        ps.addAll(ln2.parameters());
        ps.addAll(attn.parameters());
        ps.addAll(mlp.parameters());
        return ps;
    }
}

