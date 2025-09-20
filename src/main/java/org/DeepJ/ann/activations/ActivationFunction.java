package org.DeepJ.ann.activations;

import org.DeepJ.ann.Tensor;

public interface ActivationFunction {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
}
