package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.UnaryOperator;

final class NeuralOperators {
    private NeuralOperators() {}

    static void register(OnnxRegistry r) {
        unary(r, "Relu", Tensor::reluActivation);
        unary(r, "Sigmoid", Tensor::sigmoidActivation);
        unary(r, "Tanh", Tensor::tanhActivation);
        unary(r, "Softplus", NeuralOperators::softplus);
        unary(r, "Softsign", x -> x.divide(ArithmeticOperators.abs(x).addScalar(1)));
        r.add("Erf", 9, "", c -> c.one(SpecialFunctions.erf(c.tensor(0))));
        r.add("Gelu", 20, "approximate", c -> c.one(gelu(c)));
        parametric(r);
        NormalizationOperators.register(r);
    }

    private static void unary(OnnxRegistry r, String name, UnaryOperator<Tensor> operation) {
        r.add(name, 6, "", c -> c.one(operation.apply(c.tensor(0))));
    }

    static Tensor leaky(Tensor input, float alpha) {
        return input.reluActivation().subtract(input.neg().reluActivation().multiplyScalar(alpha));
    }

    static Tensor elu(Tensor input, float alpha) {
        return input.reluActivation().add(input.clamp(Float.NEGATIVE_INFINITY, 0).exp().addScalar(-1)
                                .multiplyScalar(alpha));
    }

    static Tensor softplus(Tensor input) {
        return input.reluActivation().add(ArithmeticOperators.abs(input).neg().exp().addScalar(1).log());
    }

    private static Tensor gelu(OnnxContext c) {
        String approximate = c.string("approximate", "none");
        if (approximate.equals("tanh")) return c.tensor(0).geluActivation();
        if (!approximate.equals("none"))
            throw new IllegalArgumentException("Invalid Gelu approximation");
        Tensor value = c.tensor(0).to(io.github.kirstenali.deepj.tensor.DType.FLOAT32);
        return value.multiply(SpecialFunctions.erf(value.multiplyScalar((float) (1 / Math.sqrt(2)))).addScalar(1))
                .multiplyScalar(0.5f).to(c.tensor(0).dtype());
    }

    private static Tensor clip(OnnxContext c) {
        float min = c.hasInput(1) ? c.tensor(1).item() : c.floating("min", -Float.MAX_VALUE);
        float max = c.hasInput(2) ? c.tensor(2).item() : c.floating("max", Float.MAX_VALUE);
        return c.tensor(0).clamp(min, max);
    }

    private static void parametric(OnnxRegistry r) {
        r.add("LeakyRelu", 6, "alpha", c -> c.one(leaky(c.tensor(0), c.floating("alpha", 0.01f))));
        r.add("PRelu", 7, "", c -> c.one(prelu(c.tensor(0), c.tensor(1))));
        r.add("Elu", 6, "alpha", c -> c.one(elu(c.tensor(0), c.floating("alpha", 1))));
        r.add("Selu", 6, "alpha gamma", c -> c.one(selu(c)));
        r.add("HardSigmoid", 6, "alpha beta", c -> c.one(hardSigmoid(c)));
        r.add("HardSwish", 14, "", c -> c.one(c.tensor(0).multiply(c.tensor(0).divideScalar(6).addScalar(0.5f)
                                                        .clamp(0, 1))));
        r.add("Mish", 18, "", c -> c.one(c.tensor(0).multiply(softplus(c.tensor(0)).tanhActivation())));
        r.add("Clip", 6, "min max", c -> c.one(clip(c)));
    }

    private static Tensor prelu(Tensor input, Tensor slope) {
        return input.reluActivation().subtract(input.neg().reluActivation().multiply(slope));
    }

    private static Tensor selu(OnnxContext c) {
        return elu(c.tensor(0), c.floating("alpha", 1.6732632f)).multiplyScalar(c.floating("gamma", 1.050701f));
    }

    private static Tensor hardSigmoid(OnnxContext c) {
        return c.tensor(0).multiplyScalar(c.floating("alpha", 0.2f)).addScalar(c.floating("beta", 0.5f)).clamp(0, 1);
    }
}
