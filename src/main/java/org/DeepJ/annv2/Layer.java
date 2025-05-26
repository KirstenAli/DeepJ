package org.DeepJ.annv2;

import org.DeepJ.transformer.Tensor;

public interface Layer {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput, double learningRate);
}
