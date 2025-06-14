package org.DeepJ.annv2.layers;

import org.DeepJ.annv2.Tensor;

public interface Layer {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput, double learningRate);
    default void step() {}
}
