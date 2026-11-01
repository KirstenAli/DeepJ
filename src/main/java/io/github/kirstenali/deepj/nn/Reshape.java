package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Reshapes the full tensor, with at most one inferred (-1) dimension. */
public final class Reshape extends Module {
    private final int[] shape;

    public Reshape(int... shape) {
        this.shape = shape.clone();
        inferredShape(1, false);
    }

    @Override public Tensor forward(Tensor input) { return input.reshape(inferredShape(input.numel(), true)); }

    private int[] inferredShape(int elements, boolean resolve) {
        int[] result = shape.clone();
        int missing = -1, known = 1;
        for (int axis = 0; axis < result.length; axis++) {
            if (result[axis] == -1) {
                if (missing >= 0) throw new IllegalArgumentException("Only one inferred dimension is allowed");
                missing = axis;
            } else known = Math.multiplyExact(known, NeuralChecks.positive(result[axis], "dimension"));
        }
        if (!resolve || missing < 0) return result;
        if (elements % known != 0) throw new IllegalArgumentException("Shape does not divide tensor size");
        result[missing] = elements / known;
        return result;
    }
}
