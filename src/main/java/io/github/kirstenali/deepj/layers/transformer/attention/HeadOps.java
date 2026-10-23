package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.Tensor;

final class HeadOps {

    private HeadOps() {}

    static Tensor splitHeads(Tensor tensor, int heads) {
        return LayoutOps.splitHeads(tensor, heads);
    }

    static Tensor mergeHeads(Tensor tensor, int heads) {
        return LayoutOps.mergeHeads(tensor, heads);
    }

    static Tensor dotProductScores(Tensor queries, Tensor keys, int heads) {
        return batched(queries, keys, heads, false, true);
    }

    static Tensor applyAttentionToValues(Tensor attention, Tensor values, int heads) {
        return batched(attention, values, heads, false, false);
    }

    private static Tensor batched(Tensor left, Tensor right, int heads,
                                  boolean transposeLeft, boolean transposeRight) {
        return LayoutOps.batchedMatmul(
                left, right, heads, transposeLeft, transposeRight);
    }
}
