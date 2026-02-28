package org.deepj.ann.loss;

import org.deepj.ann.Tensor;

public interface LossFunction {
    double loss(Tensor predicted, Tensor actual);
    Tensor gradient(Tensor predicted, Tensor actual);
}
