package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Saved compact values are decoded before derivative arithmetic, not repeatedly requantized. */
final class GradientDType {
    private GradientDType() {}

    static Tensor accumulation(Tensor value) {
        return Tensor.backend().cast(value, value.dtype().gradientType());
    }

    static Tensor square(Tensor value) {
        Tensor decoded = accumulation(value);
        return decoded.multiply(decoded);
    }
}
