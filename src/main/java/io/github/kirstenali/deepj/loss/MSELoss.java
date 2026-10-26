package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.autograd.GradMode;

public class MSELoss implements LossFunction {
    @Override
    public Tensor forward(Tensor predicted, Tensor actual) {
        Tensor difference = predicted.subtract(actual);
        return difference.multiply(difference).sumTensor()
                .divideScalar(predicted.matrixRows() * predicted.matrixCols());
    }

    @Override
    public float loss(Tensor predicted, Tensor actual) {
        try (var ignored = GradMode.noGrad()) {
            return forward(predicted, actual).item();
        }
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        try (var ignored = GradMode.noGrad()) {
            return predicted.subtract(actual).multiplyScalar(2.0f / (predicted.matrixRows() * predicted.matrixCols()));
        }
    }
}
