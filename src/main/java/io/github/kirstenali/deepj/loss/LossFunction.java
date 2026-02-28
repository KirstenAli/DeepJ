package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.Tensor;

public interface LossFunction {
    double loss(Tensor predicted, Tensor actual);
    Tensor gradient(Tensor predicted, Tensor actual);
}
