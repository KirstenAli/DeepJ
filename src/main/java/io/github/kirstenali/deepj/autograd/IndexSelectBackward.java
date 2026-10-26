package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Embedding-sized leaf gradients accumulate directly instead of allocating a full temporary per lookup. */
final class IndexSelectBackward implements LeafGradientRule {
    private final GradientState input;
    private final int[] shape, indices;
    private final int axis;

    IndexSelectBackward(Tensor input, int axis, int[] indices) {
        this.input = input.gradientState();
        shape = input.shape();
        this.axis = axis;
        this.indices = indices;
    }

    @Override public boolean accumulate(Tensor gradient) {
        if (input.node() != null || input.gradient() == gradient) return false;
        input.accumulateIndexed(shape, axis, indices, gradient);
        return true;
    }

    @Override public Tensor[] apply(Tensor gradient) {
        return new Tensor[]{Tensor.backend().indexSelectGradient(gradient, shape, axis, indices)};
    }
}
