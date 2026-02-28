package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.Tensor;

public interface ActivationFunction {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
}
