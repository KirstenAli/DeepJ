package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.autograd.Autograd;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

/**
 * Floating-point erf approximation composed on the selected DeepJ backend, with its analytic
 * gradient.
 */
final class SpecialFunctions {
    private SpecialFunctions() {}

    static Tensor erf(Tensor input) {
        Tensor output;
        try (var ignored = GradMode.noGrad()) {
            output = approximate(input.to(DType.FLOAT32)).to(input.dtype());
        }
        return Autograd.recordSaved(output, gradient -> new Tensor[] {derivative(input).multiply(gradient)},
                new Tensor[] {input}, input);
    }

    private static Tensor approximate(Tensor input) {
        Tensor absolute = ArithmeticOperators.abs(input), t = absolute.multiplyScalar(0.3275911f).addScalar(1).pow(-1);
        Tensor polynomial = t.multiplyScalar(1.061405429f).addScalar(-1.453152027f).multiply(t).addScalar(1.421413741f)
                        .multiply(t).addScalar(-0.284496736f).multiply(t).addScalar(0.254829592f).multiply(t);
        Tensor magnitude = polynomial.multiply(absolute.multiply(absolute).neg().exp()).neg().addScalar(1);
        return magnitude.multiply(input.clamp(-Float.MIN_NORMAL, Float.MIN_NORMAL).divideScalar(Float.MIN_NORMAL));
    }

    private static Tensor derivative(Tensor input) {
        Tensor value = input.to(DType.FLOAT32);
        return value.multiply(value).neg().exp().multiplyScalar((float) (2 / Math.sqrt(Math.PI)));
    }
}
