package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.function.UnaryOperator;

final class NeuralOperators {
    private NeuralOperators() {}

    static void register(OnnxRegistry registry) {
        unary(registry, "Relu", Tensor::reluActivation);
        unary(registry, "Sigmoid", Tensor::sigmoidActivation);
        unary(registry, "Tanh", Tensor::tanhActivation);
        unary(registry, "Softplus", NeuralOperators::softplus);
        unary(registry, "Softsign", x -> x.divide(ArithmeticOperators.abs(x).addScalar(1)));
        registry.add("Erf", 9, "", context -> context.one(SpecialFunctions.erf(context.tensor(0))));
        registry.add("Gelu", 20, "approximate", context -> context.one(gelu(context)));
        parametric(registry);
        NormalizationOperators.register(registry);
    }

    private static void unary(OnnxRegistry registry, String name, UnaryOperator<Tensor> operation) {
        registry.add(name, 6, "", context -> context.one(operation.apply(context.tensor(0))));
    }

    static Tensor leaky(Tensor input, float alpha) {
        return input.reluActivation().subtract(input.neg().reluActivation().multiplyScalar(alpha));
    }

    static Tensor elu(Tensor input, float alpha) {
        return input.reluActivation()
                .add(
                        input.clamp(Float.NEGATIVE_INFINITY, 0)
                                .exp()
                                .addScalar(-1)
                                .multiplyScalar(alpha));
    }

    static Tensor softplus(Tensor input) {
        return input.reluActivation()
                .add(ArithmeticOperators.abs(input).neg().exp().addScalar(1).log());
    }

    private static Tensor gelu(OnnxContext context) {
        String approximate = context.string("approximate", "none");
        if (approximate.equals("tanh")) return context.tensor(0).geluActivation();
        if (!approximate.equals("none"))
            throw new IllegalArgumentException("Invalid Gelu approximation");
        Tensor value = context.tensor(0).to(DType.FLOAT32);
        return value.multiply(
                        SpecialFunctions.erf(value.multiplyScalar((float) (1 / Math.sqrt(2))))
                                .addScalar(1))
                .multiplyScalar(0.5f)
                .to(context.tensor(0).dtype());
    }

    private static Tensor clip(OnnxContext context) {
        float min =
                context.hasInput(1)
                        ? context.tensor(1).item()
                        : context.floating("min", -Float.MAX_VALUE);
        float max =
                context.hasInput(2)
                        ? context.tensor(2).item()
                        : context.floating("max", Float.MAX_VALUE);
        return context.tensor(0).clamp(min, max);
    }

    private static void parametric(OnnxRegistry registry) {
        registerSignedActivations(registry);
        registry.add("HardSigmoid", 6, "alpha beta", context -> context.one(hardSigmoid(context)));
        registry.add("HardSwish", 14, "", context -> context.one(hardSwish(context.tensor(0))));
        registry.add(
                "Mish",
                18,
                "",
                context ->
                        context.one(
                                context.tensor(0)
                                        .multiply(softplus(context.tensor(0)).tanhActivation())));
        registry.add("Clip", 6, "min max", context -> context.one(clip(context)));
    }

    private static Tensor prelu(Tensor input, Tensor slope) {
        return input.reluActivation().subtract(input.neg().reluActivation().multiply(slope));
    }

    private static Tensor selu(OnnxContext context) {
        return elu(context.tensor(0), context.floating("alpha", 1.6732632f))
                .multiplyScalar(context.floating("gamma", 1.050701f));
    }

    private static Tensor hardSigmoid(OnnxContext context) {
        return context.tensor(0)
                .multiplyScalar(context.floating("alpha", 0.2f))
                .addScalar(context.floating("beta", 0.5f))
                .clamp(0, 1);
    }

    private static Tensor hardSwish(Tensor input) {
        return input.multiply(input.divideScalar(6).addScalar(0.5f).clamp(0, 1));
    }

    private static void registerSignedActivations(OnnxRegistry registry) {
        registry.add(
                "LeakyRelu",
                6,
                "alpha",
                context -> context.one(leaky(context.tensor(0), context.floating("alpha", 0.01f))));
        registry.add(
                "PRelu",
                7,
                "",
                context -> context.one(prelu(context.tensor(0), context.tensor(1))));
        registry.add(
                "Elu",
                6,
                "alpha",
                context -> context.one(elu(context.tensor(0), context.floating("alpha", 1))));
        registry.add("Selu", 6, "alpha gamma", context -> context.one(selu(context)));
    }
}
