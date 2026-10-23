package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class ActivationOps {
    private ActivationOps() {}

    public static Tensor tanh(Tensor input) {
        Tensor output = Tensor.backend().tanh(input);
        return UnaryOps.record(input, output, g -> g.multiply(output.multiply(output).neg().addScalar(1)), output);
    }

    public static Tensor sigmoid(Tensor input) {
        Tensor output = Tensor.backend().sigmoid(input);
        return UnaryOps.record(input, output, g -> g.multiply(output).multiply(output.neg().addScalar(1)), output);
    }

    public static Tensor relu(Tensor input) {
        return UnaryOps.record(input, Tensor.backend().relu(input),
                g -> Tensor.backend().reluBackward(input, g), input);
    }

    public static Tensor gelu(Tensor input) {
        return UnaryOps.record(input, Tensor.backend().gelu(input),
                g -> Tensor.backend().geluBackward(input, g), input);
    }

    public static Tensor softmax(Tensor input) {
        Tensor output = Tensor.backend().softmaxRows(input);
        return UnaryOps.record(input, output, g -> Tensor.backend().softmaxBackward(g, output), output);
    }

    public static Tensor causalSoftmax(Tensor input, int length, float scale) {
        Tensor output = Tensor.backend().causalSoftmax(input, length, scale);
        return UnaryOps.record(input, output,
                g -> Tensor.backend().softmaxBackward(g, output, scale), output);
    }

    public static Tensor swiGlu(Tensor gate, Tensor up) {
        Tensor output = Tensor.backend().swiGlu(gate, up);
        return Autograd.record(output, g -> {
            var result = Tensor.backend().swiGluBackward(g, gate, up);
            return new Tensor[]{result.gateGradient(), result.upGradient()};
        }, gate, up);
    }
}
