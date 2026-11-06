package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

final class RecurrentActivation {
    private RecurrentActivation() {}

    static Tensor apply(String name, Tensor input, Float alpha, Float beta) {
        return switch (name) {
            case "Relu" -> input.reluActivation();
            case "Tanh" -> input.tanhActivation();
            case "Sigmoid" -> input.sigmoidActivation();
            case "Affine" -> input.multiplyScalar(value(alpha, 1)).addScalar(value(beta, 0));
            case "LeakyRelu" -> NeuralOperators.leaky(input, value(alpha, 0.01f));
            case "ScaledTanh" ->
                    input.multiplyScalar(value(beta, 1))
                            .tanhActivation()
                            .multiplyScalar(value(alpha, 1));
            default -> extended(name, input, alpha, beta);
        };
    }

    private static float value(Float value, float fallback) {
        return value == null ? fallback : value;
    }

    private static Tensor extended(String name, Tensor input, Float alpha, Float beta) {
        return switch (name) {
            case "HardSigmoid" ->
                    input.multiplyScalar(value(alpha, 0.2f))
                            .addScalar(value(beta, 0.5f))
                            .clamp(0, 1);
            case "Elu" -> NeuralOperators.elu(input, value(alpha, 1));
            case "Softsign" -> input.divide(ArithmeticOperators.abs(input).addScalar(1));
            case "Softplus" -> NeuralOperators.softplus(input);
            case "ThresholdedRelu" -> thresholded(input, value(alpha, 1));
            default -> throw new IllegalArgumentException("Unsupported RNN activation " + name);
        };
    }

    private static Tensor thresholded(Tensor input, float alpha) {
        Tensor cutoff = Tensor.from(new float[] {alpha}, input.dtype(), new int[0]);
        return BooleanOperators.where(
                BooleanOperators.compare(input, cutoff, "Greater"),
                input,
                Tensor.backend().zeros(input.dtype(), input.shape()));
    }
}
