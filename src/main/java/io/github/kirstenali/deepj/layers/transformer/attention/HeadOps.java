package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.tensor.Tensor;

final class HeadOps {

    private HeadOps() {}

    record AttentionGrads(Tensor dScores, Tensor dVh) {}
    record QKGrads(Tensor dQh, Tensor dKh) {}

    static Tensor splitHeads(Tensor tensor, int heads) {
        return Tensor.backend().splitHeads(tensor, heads);
    }

    static Tensor mergeHeads(Tensor tensor, int heads) {
        return Tensor.backend().mergeHeads(tensor, heads);
    }

    static Tensor dotProductScores(Tensor queries, Tensor keys, int heads) {
        return batched(queries, keys, heads, false, true);
    }

    static Tensor applyAttentionToValues(Tensor attention, Tensor values, int heads) {
        return batched(attention, values, heads, false, false);
    }

    static AttentionGrads backwardAttentionAndValues(
            Tensor outputGradient, Tensor values, Tensor attention,
            ActivationFunction softmax, float scale, int heads) {
        Tensor attentionGradient = batched(outputGradient, values, heads, false, true);
        Tensor valueGradient = batched(attention, outputGradient, heads, true, false);
        Tensor scoreGradient = softmax.backward(attentionGradient).multiplyScalar(scale);
        return new AttentionGrads(scoreGradient, valueGradient);
    }

    static QKGrads backwardQueriesAndKeys(
            Tensor scoreGradient, Tensor queries, Tensor keys, int heads) {
        Tensor queryGradient = batched(scoreGradient, keys, heads, false, false);
        Tensor keyGradient = batched(scoreGradient, queries, heads, true, false);
        return new QKGrads(queryGradient, keyGradient);
    }

    private static Tensor batched(Tensor left, Tensor right, int heads,
                                  boolean transposeLeft, boolean transposeRight) {
        return Tensor.backend().batchedMatmul(
                left, right, heads, transposeLeft, transposeRight);
    }
}
