package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

final class MetalPerformanceWorkloads {

    private MetalPerformanceWorkloads() {}

    static Tensor mixedTen(TensorBackend backend, Tensor left, Tensor right) {
        Tensor value = backend.matmul(left, right);
        value = backend.gelu(value);
        value = backend.multiplyScalar(value, 0.5f);
        value = backend.subtract(value, right);
        value = backend.relu(value);
        value = backend.matmul(value, left);
        value = backend.sigmoid(value);
        value = backend.multiply(value, right);
        value = backend.tanh(value);
        return backend.neg(value);
    }

    static Tensor mixedChain(TensorBackend backend, Tensor left, Tensor right, int operations) {
        Tensor value = backend.matmul(left, right);
        for (int index = 0; index < operations; index++) {
            value = applyMixedOperation(backend, left, right, value, index);
        }
        return value;
    }

    private static Tensor applyMixedOperation(TensorBackend backend, Tensor left,
                                              Tensor right, Tensor value, int index) {
        return switch (index % 5) {
            case 0 -> backend.gelu(value);
            case 1 -> backend.matmul(value, left);
            case 2 -> backend.add(value, right);
            case 3 -> backend.sigmoid(value);
            default -> backend.subtract(value, left);
        };
    }

    static Tensor backward(TensorBackend backend, Tensor gradient, Tensor softmax,
                           Tensor w1, Tensor w2, Tensor preActivation, Tensor offset) {
        Tensor value = backend.softmaxBackward(gradient, softmax);
        value = backend.matmul(value, w2);
        value = backend.geluBackward(preActivation, value);
        value = backend.matmul(value, w1);
        value = backend.subtract(value, offset);
        return backend.multiplyScalar(value, 0.5f);
    }
}
