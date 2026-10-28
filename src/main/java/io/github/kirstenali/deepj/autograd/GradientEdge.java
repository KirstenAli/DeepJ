package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** A graph connection must not keep an otherwise unused activation's storage alive. */
record GradientEdge(GradientState state, int[] shape, long version) {
    GradientEdge(Tensor tensor) {
        this(tensor.gradientState(), tensor.shape(), tensor.gradientState().version());
    }

    @Override public int[] shape() { return shape.clone(); }

    void validate() {
        if (state.version() != version) {
            throw new IllegalStateException("A graph input was modified after forward");
        }
    }

    void requireShape(Tensor gradient) {
        if (!gradient.hasShape(shape)) {
            throw new IllegalArgumentException("Operation gradient must match its input shape");
        }
    }
}
