package io.github.kirstenali.deepj.tensor;

/** Optional backend capability for memory-efficient, fused causal attention. */
public interface CausalAttentionBackend extends TensorBackend {
    boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale);

    CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value, int heads, float scale);

    AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                              CausalAttentionResult saved, Tensor gradient,
                                              int heads, float scale);

    default boolean supportsCausalAttention(Tensor query, Tensor key, Tensor value,
                                            int queryHeads, int keyValueHeads, float scale) {
        return queryHeads == keyValueHeads && supportsCausalAttention(query, key, value, queryHeads, scale);
    }

    default CausalAttentionResult causalAttention(Tensor query, Tensor key, Tensor value,
                                                 int queryHeads, int keyValueHeads, float scale) {
        requireSharedHeads(queryHeads, keyValueHeads);
        return causalAttention(query, key, value, queryHeads, scale);
    }

    default AttentionGradients causalAttentionBackward(Tensor query, Tensor key, Tensor value,
                                                       CausalAttentionResult saved, Tensor gradient,
                                                       int queryHeads, int keyValueHeads, float scale) {
        requireSharedHeads(queryHeads, keyValueHeads);
        return causalAttentionBackward(query, key, value, saved, gradient, queryHeads, scale);
    }

    private static void requireSharedHeads(int queryHeads, int keyValueHeads) {
        if (queryHeads != keyValueHeads) {
            throw new UnsupportedOperationException("This backend does not support compact grouped attention");
        }
    }
}
