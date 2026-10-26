package io.github.kirstenali.deepj.tensor;

/** Explicit compatibility for packed matrix APIs,delegating their work to the same ND operators. */
public final class PackedLayouts {
    private PackedLayouts() {}

    public static Tensor matmul(TensorBackend backend, Tensor left, Tensor right, int batches,
                                 boolean transposeLeft, boolean transposeRight) {
        var shape = PackedMatmulShape.of(left, right, batches, transposeLeft, transposeRight);
        Tensor a = operand(backend, left, batches, transposeLeft);
        Tensor b = operand(backend, right, batches, transposeRight);
        return backend.reshape(backend.matmul(a, b), batches * shape.rows(), shape.cols());
    }

    private static Tensor operand(TensorBackend backend, Tensor input, int batches, boolean transpose) {
        Tensor matrix = backend.reshape(input, batches, input.matrixRows() / batches, input.matrixCols());
        return transpose ? backend.permute(matrix, 0, 2, 1) : matrix;
    }

    public static Tensor split(TensorBackend backend, Tensor input, int heads) {
        requireDivisible(input.matrixCols(), heads);
        int time = input.matrixRows(), width = input.matrixCols() / heads;
        Tensor grouped = backend.reshape(input, time, heads, width);
        return backend.reshape(backend.permute(grouped, 1, 0, 2), heads * time, width);
    }

    public static Tensor merge(TensorBackend backend, Tensor input, int heads) {
        requireDivisible(input.matrixRows(), heads);
        int time = input.matrixRows() / heads, width = input.matrixCols();
        Tensor grouped = backend.reshape(input, heads, time, width);
        return backend.reshape(backend.permute(grouped, 1, 0, 2), time, heads * width);
    }

    private static void requireDivisible(int dimension, int heads) {
        if (heads < 1 || dimension % heads != 0) {
            throw new IllegalArgumentException("Packed dimensions must be divisible by a positive head count");
        }
    }
}
