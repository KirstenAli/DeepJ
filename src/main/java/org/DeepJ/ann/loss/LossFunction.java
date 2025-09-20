package org.DeepJ.ann.loss;

import org.DeepJ.ann.Tensor;

public interface LossFunction {
    double loss(Tensor predicted, Tensor actual);
    Tensor gradient(Tensor predicted, Tensor actual);
}
