package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Objects;

/** Adapts DeepJ activations to the existing Layer contract. */
public final class Activation extends Module {
    private final ActivationFunction function;

    public Activation(ActivationFunction function) { this.function = Objects.requireNonNull(function, "function"); }

    public static Activation relu() { return new Activation(Tensor::reluActivation); }
    public static Activation gelu() { return new Activation(Tensor::geluActivation); }
    public static Activation sigmoid() { return new Activation(Tensor::sigmoidActivation); }
    public static Activation tanh() { return new Activation(Tensor::tanhActivation); }
    public static Activation softmax(int axis) { return new Activation(input -> input.softmax(axis)); }

    @Override public Tensor forward(Tensor input) { return function.forward(input); }
}
