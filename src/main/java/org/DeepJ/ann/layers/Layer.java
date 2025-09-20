package org.DeepJ.ann.layers;

import org.DeepJ.ann.Tensor;

public interface Layer {
    Tensor forward(Tensor input);
    Tensor backward(Tensor gradOutput, double learningRate);
    default void step() {}
}
