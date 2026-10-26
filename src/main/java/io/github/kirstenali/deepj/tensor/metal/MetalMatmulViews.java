package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;

/** The checked legacy batched-matrix API delegates general layouts to the same ND GPU operator. */
final class MetalMatmulViews {
    private MetalMatmulViews() {}

    static Tensor batched(MetalCoreOps backend, Tensor left, Tensor right, int batches,
                          boolean transposeLeft, boolean transposeRight) {
        var shape = MetalCoreOps.batchedShape(left, right, batches, transposeLeft, transposeRight);
        Tensor a = operand(backend, left, batches, transposeLeft);
        Tensor b = operand(backend, right, batches, transposeRight);
        return backend.reshape(backend.matmul(a, b), batches * shape.rows(), shape.cols());
    }

    private static Tensor operand(MetalCoreOps backend, Tensor input, int batches, boolean transpose) {
        Tensor matrix = backend.reshape(input, batches, input.matrixRows() / batches, input.matrixCols());
        return transpose ? backend.permute(matrix, 0, 2, 1) : matrix;
    }
}
