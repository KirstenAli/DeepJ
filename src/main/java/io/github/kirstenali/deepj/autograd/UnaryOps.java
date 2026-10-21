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
        return record(input, output, g -> g.divide(output).multiplyScalar(0.5f));
    }

    public static Tensor neg(Tensor input) {
        return record(input, Tensor.backend().neg(input), Tensor::neg);
    }

    public static Tensor exp(Tensor input) {
        Tensor output = Tensor.backend().exp(input);
        return record(input, output, g -> g.multiply(output));
    }

    public static Tensor log(Tensor input) {
        return record(input, Tensor.backend().log(input), g -> g.divide(input));
    }

    public static Tensor pow(Tensor input, float exponent) {
        return record(input, Tensor.backend().pow(input, exponent),
                g -> exponent == 0 ? Tensor.zeros(input.rows, input.cols)
                        : g.multiply(input.pow(exponent - 1)).multiplyScalar(exponent));
    }

    static Tensor record(Tensor input, Tensor output, Function<Tensor, Tensor> derivative) {
        return Autograd.recordSaved(output, g -> new Tensor[]{derivative.apply(g)},
                new Tensor[]{input}, input, output);
    }
}
