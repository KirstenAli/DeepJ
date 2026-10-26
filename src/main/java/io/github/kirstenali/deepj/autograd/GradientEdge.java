package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** A graph connection must not keep an otherwise unused activation's storage alive. */
record GradientEdge(GradientState state, int rows, int cols, long version) {
    GradientEdge(Tensor tensor) {
        this(tensor.gradientState(), tensor.matrixRows(), tensor.matrixCols(), tensor.gradientState().version());
    }

    void validate() {
        if (state.version() != version) {
            throw new IllegalStateException("A graph input was modified after forward");
        }
    }

    void requireShape(Tensor gradient) {
        if (gradient.matrixRows() != rows || gradient.matrixCols() != cols) {
            throw new IllegalArgumentException("Operation gradient must match its input shape");
        }
    }
}
