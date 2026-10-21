package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class LayoutOps {
    private LayoutOps() {}

    public static Tensor gather(Tensor input, int[] indices) {
        int[] saved = indices.clone();
        Tensor output = Tensor.backend().sliceRows(input, saved);
        return Autograd.record(output, g -> new Tensor[]{scatter(input, saved, g)}, input);
    }

    private static Tensor scatter(Tensor input, int[] indices, Tensor gradient) {
        Tensor result = Tensor.zeros(input.rows, input.cols);
        Tensor.backend().scatterAddRows(result, indices, gradient);
        return result;
    }

    public static Tensor splitHeads(Tensor input, int heads) {
        return UnaryOps.record(input, Tensor.backend().splitHeads(input, heads),
                g -> Tensor.backend().mergeHeads(g, heads));
    }

    public static Tensor mergeHeads(Tensor input, int heads) {
        return UnaryOps.record(input, Tensor.backend().mergeHeads(input, heads),
                g -> Tensor.backend().splitHeads(g, heads));
    }

    public static Tensor batchedMatmul(Tensor a, Tensor b, int batches, boolean ta, boolean tb) {
        Tensor output = batch(a, b, batches, ta, tb);
        return BinaryOps.record(output, a, b,
                g -> ta ? batch(b, g, batches, tb, true) : batch(g, b, batches, false, !tb),
                g -> tb ? batch(g, a, batches, true, ta) : batch(a, g, batches, !ta, false));
    }

    private static Tensor batch(Tensor a, Tensor b, int count, boolean ta, boolean tb) {
        return Tensor.backend().batchedMatmul(a, b, count, ta, tb);
    }

    public static Tensor rotary(Tensor input, Tensor cosine, Tensor sine, int length, boolean inverse) {
        if (cosine.requiresGrad() || sine.requiresGrad()) {
            throw new IllegalArgumentException("Rotary tables must be constants");
        }
        Tensor output = Tensor.backend().rotary(input, cosine, sine, length, inverse);
        return Autograd.recordSaved(output,
                g -> new Tensor[]{Tensor.backend().rotary(g, cosine, sine, length, !inverse)},
                new Tensor[]{input}, input, cosine, sine);
    }
}
