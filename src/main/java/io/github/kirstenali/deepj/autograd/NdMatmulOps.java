package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.stream.IntStream;

final class NdMatmulOps {
    private NdMatmulOps() {}

    static Tensor apply(Tensor a, Tensor b) {
        return BinaryOps.record(Tensor.backend().matmul(a, b), a, b,
                gradient -> derivative(a, b, gradient, true),
                gradient -> derivative(a, b, gradient, false));
    }

    private static Tensor derivative(Tensor a, Tensor b, Tensor gradient, boolean left) {
        Tensor first = promoted(a, 0), second = promoted(b, 1);
        gradient = upstream(gradient, a.rank() == 1, b.rank() == 1);
        Tensor product = left ? Tensor.backend().matmul(gradient, transpose(second))
                : Tensor.backend().matmul(transpose(first), gradient);
        int[] matrixShape = left ? first.shape() : second.shape();
        int[] original = left ? a.shape() : b.shape();
        return Tensor.backend().reshape(Tensor.backend().sumToShape(product, matrixShape), original);
    }

    private static Tensor promoted(Tensor input, int axis) {
        return input.rank() == 1 ? Tensor.backend().unsqueeze(input, axis) : input;
    }

    private static Tensor upstream(Tensor gradient, boolean leftVector, boolean rightVector) {
        if (rightVector) gradient = Tensor.backend().unsqueeze(gradient, -1);
        if (leftVector) gradient = Tensor.backend().unsqueeze(gradient, gradient.rank() - 1);
        return gradient;
    }

    private static Tensor transpose(Tensor input) {
        int[] axes = IntStream.range(0, input.rank()).toArray();
        axes[axes.length - 2] = axes.length - 1;
        axes[axes.length - 1] = axes.length - 2;
        return Tensor.backend().permute(input, axes);
    }
}
