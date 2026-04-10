package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

public class MSELoss implements LossFunction {
    @Override
    public float loss(Tensor predicted, Tensor actual) {
        Tensor diff = predicted.subtract(actual);
        Tensor sq = diff.multiply(diff);
        float sse = sq.sum();
        return sse / (predicted.rows * predicted.cols);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        Tensor grad = predicted.subtract(actual);
        grad.multiplyScalarInPlace(2.0f / (predicted.rows * predicted.cols));
        return grad;
    }
}