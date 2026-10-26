package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

public final class UnaryOps {
    private UnaryOps() {}

    public static Tensor scale(Tensor input, float value) {
        return record(input, Tensor.backend().multiplyScalar(input, value), g -> g.multiplyScalar(value));
    }

    public static Tensor shift(Tensor input, float value) {
        return record(input, Tensor.backend().addScalar(input, value), g -> g);
    }

    public static Tensor divide(Tensor input, float value) {
        return record(input, Tensor.backend().divideScalar(input, value), g -> g.divideScalar(value));
    }

    public static Tensor transpose(Tensor input) {
        return record(input, Tensor.backend().transpose(input), Tensor::transpose);
    }

    public static Tensor sqrt(Tensor input) {
        Tensor output = Tensor.backend().sqrt(input);
        return record(input, output, g -> g.divide(output).multiplyScalar(0.5f), output);
    }

    public static Tensor neg(Tensor input) {
        return record(input, Tensor.backend().neg(input), Tensor::neg);
    }

    public static Tensor exp(Tensor input) {
        Tensor output = Tensor.backend().exp(input);
        return record(input, output, g -> g.multiply(output), output);
    }

    public static Tensor log(Tensor input) {
        return record(input, Tensor.backend().log(input), g -> g.divide(input), input);
    }

    public static Tensor pow(Tensor input, float exponent) {
        int[] shape = input.shape();
        var gradientType = input.dtype().gradientType();
        if (exponent == 0) {
            return record(input, Tensor.backend().pow(input, exponent),
                    g -> Tensor.backend().zeros(gradientType, shape));
        }
        return record(input, Tensor.backend().pow(input, exponent),
                g -> g.multiply(GradientDType.accumulation(input).pow(exponent - 1)).multiplyScalar(exponent), input);
    }

    static Tensor record(Tensor input, Tensor output, Function<Tensor, Tensor> derivative, Tensor... saved) {
        return Autograd.recordBuiltinSaved(output, g -> new Tensor[]{derivative.apply(g)},
                new Tensor[]{input}, saved);
    }
}
