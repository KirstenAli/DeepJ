package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public class ReLU implements ActivationFunction {
    private Tensor input;

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        return input.reluActivation();
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        return input.reluBackward(gradOutput);
    }
}
