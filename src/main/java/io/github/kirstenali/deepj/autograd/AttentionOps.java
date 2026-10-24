package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.CausalAttentionBackend;
import io.github.kirstenali.deepj.tensor.Tensor;

public final class AttentionOps {
    private AttentionOps() {}

    public static Tensor causal(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        return causal(query, key, value, heads, heads, scale);
    }

    public static Tensor causal(Tensor query, Tensor key, Tensor value,
                                 int queryHeads, int keyValueHeads, float scale) {
        requireLayout(query, key, value, queryHeads, keyValueHeads, scale);
        if (Tensor.backend() instanceof CausalAttentionBackend backend
                && backend.supportsCausalAttention(query, key, value, queryHeads, keyValueHeads, scale)) {
            return fused(backend, query, key, value, queryHeads, keyValueHeads, scale);
        }
        Tensor scores = LayoutOps.batchedMatmul(query, key, keyValueHeads, false, true);
        Tensor probabilities = ActivationOps.causalSoftmax(scores, query.rows / queryHeads, scale);
        return LayoutOps.batchedMatmul(probabilities, value, keyValueHeads, false, false);
    }

    private static void requireLayout(Tensor query, Tensor key, Tensor value,
                                       int queryHeads, int keyValueHeads, float scale) {
        requireHeads(queryHeads, keyValueHeads);
        if (query.rows % queryHeads != 0 || key.rows % keyValueHeads != 0
                || query.rows / queryHeads != key.rows / keyValueHeads
                || query.cols != key.cols || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("Invalid causal attention heads, shape or scale");
        }
        Tensor.requireSameShape(key, value, "causalAttention");
    }

    private static void requireHeads(int queryHeads, int keyValueHeads) {
        if (queryHeads < 1 || keyValueHeads < 1 || queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("Query heads must be a positive multiple of key/value heads");
        }
    }

    private static Tensor fused(CausalAttentionBackend backend, Tensor query, Tensor key,
                                 Tensor value, int queryHeads, int keyValueHeads, float scale) {
        var saved = backend.causalAttention(query, key, value, queryHeads, keyValueHeads, scale);
        return Autograd.recordSaved(saved.output(), gradient -> {
            var result = backend.causalAttentionBackward(query, key, value, saved, gradient,
                    queryHeads, keyValueHeads, scale);
            return new Tensor[]{result.query(), result.key(), result.value()};
        }, new Tensor[]{query, key, value}, query, key, value, saved.output(), saved.normalization());
    }
}
