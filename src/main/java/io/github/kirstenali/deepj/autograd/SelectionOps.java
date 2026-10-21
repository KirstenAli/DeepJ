package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class SelectionOps {
    private SelectionOps() {}

    /** Tied maxima share the row gradient equally. */
    public static Tensor maximum(Tensor input) {
        Tensor output = Tensor.backend().maxAlongRows(input);
        return UnaryOps.record(input, output, g -> maximumGradient(input, output, g), input, output);
    }

    private static Tensor maximumGradient(Tensor input, Tensor maximum, Tensor gradient) {
        Tensor ones = Tensor.ones(input.rows, input.cols);
        Tensor distance = input.subtractBroadcastCols(maximum).neg();
        Tensor selected = ones.subtract(Tensor.backend().reluBackward(distance, ones));
        Tensor shared = gradient.divide(selected.sumAlongRows());
        return selected.multiplyBroadcastCols(shared);
    }

    /** The derivative is zero at either clipping boundary. */
    public static Tensor clamp(Tensor input, float minimum, float maximum) {
        Tensor output = Tensor.backend().clamp(input, minimum, maximum);
        return UnaryOps.record(input, output, g -> clippedGradient(input, g, minimum, maximum), input);
    }

    private static Tensor clippedGradient(Tensor input, Tensor gradient, float minimum, float maximum) {
        Tensor above = Tensor.backend().reluBackward(input.addScalar(-minimum), gradient);
        return Tensor.backend().reluBackward(input.neg().addScalar(maximum), above);
    }
}
