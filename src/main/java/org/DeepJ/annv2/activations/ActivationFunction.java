package org.DeepJ.annv2.activations;

import org.DeepJ.transformer.Tensor;

public interface ActivationFunction {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput);
}
