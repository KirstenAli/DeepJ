package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.Random;

/**
 * Multi-head self-attention with Rotary Positional Embedding (RoPE).
 *
 * <p>Used by Llama, Mistral, Qwen, DeepSeek, and GPT-NeoX. Extends
 * {@link MultiHeadSelfAttention} and overrides only the two Q/K transform hooks —
 * all attention mechanics, causal masking, and backpropagation are inherited unchanged.
 *
 * <p>The {@link RotaryEmbedding} is applied to Q and K <em>after</em> they are projected
 * and split into per-head blocks, before the scaled dot-product is computed:
 * <pre>
 *   Q_rot = RoPE(Q_heads)   K_rot = RoPE(K_heads)
 *   scores = softmax( Q_rot · K_rotᵀ / √d ) · V
 * </pre>
 *
 * <p>Typical usage inside a Llama-style block:
 * <pre>{@code
 * RotaryEmbedding rope = new RotaryEmbedding(dModel / nHeads, maxSeqLen);
 * new TransformerBlock(
 *     new RMSNorm1D(dModel),
 *     new RMSNorm1D(dModel),
 *     new RoPEMultiHeadSelfAttention(dModel, nHeads, true, rope, rnd),
 *     new SwiGLULayer(dModel, dFF, rnd)
 * );
 * }</pre>
 */
public final class RoPEMultiHeadSelfAttention extends MultiHeadSelfAttention {

    private final RotaryEmbedding rope;

    /**
     * @param dModel      model (embedding) dimension
     * @param nHeads      number of attention heads; must divide {@code dModel} evenly
     * @param causalMask  {@code true} for autoregressive (decoder) attention
     * @param rope        pre-built rotary embedding sized for {@code dModel / nHeads}
     * @param rnd         random source for weight initialisation
     */
    public RoPEMultiHeadSelfAttention(int dModel, int nHeads, boolean causalMask,
                                      RotaryEmbedding rope, Random rnd) {
        super(dModel, nHeads, causalMask, rnd);
        if (rope == null) throw new IllegalArgumentException("rope must not be null");
        this.rope = rope;
    }

    /**
     * Applies RoPE rotation to Q or K heads in the forward pass.
     * Input/output shape: {@code [nHeads·seqLen × headDim]}.
     */
    @Override
    protected Tensor transformQueryKey(Tensor heads, int seqLen) {
        return rope.apply(heads, seqLen, nHeads);
    }

    /**
     * Applies the inverse (transpose) RoPE rotation to Q/K gradients in the backward pass.
     * Because rotation matrices are orthogonal, the inverse is the transpose (negate sin).
     */
    @Override
    protected Tensor transformQueryKeyBackward(Tensor gradHeads, int seqLen) {
        return rope.applyBackward(gradHeads, seqLen, nHeads);
    }
}

