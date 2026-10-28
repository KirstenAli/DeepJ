package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.autograd.GradMode;
import java.util.Arrays;

/** Mean squared error over equal, nonempty shapes, with FLOAT32 loss and gradients. */
public class MSELoss implements LossFunction {
    @Override
    public Tensor forward(Tensor predicted, Tensor actual) {
        try (var scope = TensorScope.open()) {
            Tensor difference = difference(predicted, actual);
            return scope.escape(difference.multiply(difference).sumTensor().divideScalar(predicted.numel()));
        }
    }

    @Override
    public float loss(Tensor predicted, Tensor actual) {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            return forward(predicted, actual).item();
        }
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor actual) {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            return scope.escape(difference(predicted, actual).multiplyScalar(2.0f / predicted.numel()));
        }
    }

    private static Tensor difference(Tensor predicted, Tensor actual) {
        if (predicted == null || actual == null || predicted.numel() == 0
                || !Arrays.equals(predicted.shape(), actual.shape())) {
            throw new IllegalArgumentException("MSE requires equal, nonempty tensor shapes");
        }
        return predicted.to(DType.FLOAT32).subtract(actual.to(DType.FLOAT32));
    }
}
