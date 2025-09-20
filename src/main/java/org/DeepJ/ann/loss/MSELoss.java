package org.DeepJ.ann.loss;

import org.DeepJ.ann.Tensor;

public class MSELoss implements LossFunction {
    @Override
    public double loss(Tensor predicted, Tensor actual) {
        return predicted.mseLoss(actual);
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        return predicted.subtract(actual).multiplyScalar(2.0 / (predicted.rows * predicted.cols));
    }
}