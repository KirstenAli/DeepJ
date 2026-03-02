package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;

public class MSELoss implements LossFunction {
    @Override
    public double loss(Tensor predicted, Tensor actual) {
        Tensor diff = predicted.subtract(actual);
        Tensor sq = diff.multiply(diff);
        double sse = sq.sum();
        return sse / (predicted.rows * predicted.cols);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        return predicted.subtract(actual).multiplyScalar(2.0 / (predicted.rows * predicted.cols));
    }
}