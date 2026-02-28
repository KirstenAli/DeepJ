package org.deepj.ann.activations;

import org.deepj.ann.Tensor;

public interface ActivationFunction {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
}
