package io.github.kirstenali.deepj.activations;

import io.github.kirstenali.deepj.tensor.Tensor;

public interface ActivationFunction {
    Tensor forward(Tensor input);
}
