package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public final class SiLU implements ActivationFunction {
    @Override
    public Tensor forward(Tensor input) {
        return input.multiply(input.sigmoidActivation());
    }
}
