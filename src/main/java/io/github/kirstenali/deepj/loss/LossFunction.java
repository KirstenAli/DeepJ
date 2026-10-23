package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

public interface LossFunction {
    Tensor forward(Tensor predicted, Tensor actual);
    float loss(Tensor predicted, Tensor actual);
    Tensor gradient(Tensor predicted, Tensor actual);
}
