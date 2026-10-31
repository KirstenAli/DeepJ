package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.HeadLayout;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Folding independent batch/head groups into a native invocation never mixes sequences or expands KV heads. */
final class NdAttentionOps {
    private NdAttentionOps() {}

    static Tensor causal(Tensor query, Tensor key, Tensor value, int heads, int kvHeads, float scale) {
        require(query, key, value, heads, kvHeads);
        Tensor output = AttentionOps.causal(HeadLayout.pack(query), HeadLayout.pack(key), HeadLayout.pack(value),
                HeadLayout.totalHeads(query, heads), HeadLayout.totalHeads(key, kvHeads), scale);
        return output.reshape(query.shape());
    }

    private static void require(Tensor query, Tensor key, Tensor value, int heads, int kvHeads) {
        HeadLayout.require(query, heads);
        HeadLayout.require(key, kvHeads);
        Tensor.requireSameShape(key, value, "Causal attention values");
        if (query.size(0) < 1 || query.size(2) < 1 || query.size(0) != key.size(0)
                || query.size(2) != key.size(2) || query.size(3) != key.size(3)) {
            throw new IllegalArgumentException("Causal attention requires matching nonempty batch,time and width");
        }
    }
}
