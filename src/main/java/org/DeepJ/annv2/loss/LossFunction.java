package org.DeepJ.annv2.loss;

import org.DeepJ.transformer.Tensor;

public interface LossFunction {
    double loss(Tensor predicted, Tensor actual);
    Tensor gradient(Tensor predicted, Tensor actual);
}
