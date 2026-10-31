package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Positionwise norms keep the established matrix arithmetic without discarding public batch/time axes. */
final class PositionNormalization {
    private PositionNormalization() {}

    static Tensor matrix(Tensor input, int width) {
        if (input.rank() == 0 || input.size(-1) != width) {
            throw new IllegalArgumentException("Expected final axis width=" + width);
        }
        return input.rank() == 2 ? input : input.reshape(input.numel() / width, width);
    }

    static Tensor restore(Tensor input, Tensor result) {
        return input.rank() == 2 ? result : result.reshape(input.shape());
    }
}
