package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class Tanh implements ActivationFunction {
    @Override
    public Tensor forward(Tensor input) {
        return input.tanhActivation();
    }
}
