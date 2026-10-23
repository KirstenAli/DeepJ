package io.github.kirstenali.deepj.tensor;

/** Optional backend capability for memory-efficient, fused causal attention. */
public interface CausalAttentionBackend extends TensorBackend {
    boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale);

    CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale);

    AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                              CausalAttentionResult saved, Tensor gradient,
                                              int heads, float scale);
}
