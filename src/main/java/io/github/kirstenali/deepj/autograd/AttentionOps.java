package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.CausalAttentionBackend;
import io.github.kirstenali.deepj.tensor.Tensor;

public final class AttentionOps {
    private AttentionOps() {}

    public static Tensor causal(Tensor query, Tensor key, Tensor value, int heads, float scale) {
        if (Tensor.backend() instanceof CausalAttentionBackend backend
                && backend.supportsCausalAttention(query, key, value, heads, scale)) {
            return fused(backend, query, key, value, heads, scale);
        }
        Tensor scores = LayoutOps.batchedMatmul(query, key, heads, false, true);
        Tensor probabilities = ActivationOps.causalSoftmax(scores, query.rows / heads, scale);
        return LayoutOps.batchedMatmul(probabilities, value, heads, false, false);
    }

    private static Tensor fused(CausalAttentionBackend backend, Tensor query, Tensor key,
                                 Tensor value, int heads, float scale) {
        var saved = backend.causalAttention(query, key, value, heads, scale);
        return Autograd.recordSaved(saved.output(), gradient -> {
            var result = backend.causalAttentionBackward(query, key, value, saved, gradient, heads, scale);
            return new Tensor[]{result.query(), result.key(), result.value()};
        }, new Tensor[]{query, key, value}, query, key, value, saved.output(), saved.normalization());
    }
}
