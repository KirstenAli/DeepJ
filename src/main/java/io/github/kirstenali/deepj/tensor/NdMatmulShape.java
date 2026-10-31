package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

/** Matrix promotion and trailing batch broadcasting; singleton promotion never copies values. */
final class NdMatmulShape {
    private final TensorLayout left, right;
    private final TensorShape output;
    private final int rows, inner, columns;

    NdMatmulShape(Tensor a, Tensor b) {
        if (a.rank() == 0 || b.rank() == 0) throw new IllegalArgumentException("Matmul does not accept scalars");
        TensorLayout first = a.rank() == 1 ? LayoutTransforms.unsqueeze(a.layout(), 0) : a.layout();
        TensorLayout second = b.rank() == 1 ? LayoutTransforms.unsqueeze(b.layout(), 1) : b.layout();
        rows = first.shape().size(-2);
        inner = first.shape().size(-1);
        columns = second.shape().size(-1);
        if (inner != second.shape().size(-2)) throw new IllegalArgumentException("Matmul inner dimensions must match");
        TensorShape batch = ShapeTransforms.broadcast(prefix(first.shape()), prefix(second.shape()));
        left = LayoutTransforms.expand(first, append(batch, rows, inner));
        right = LayoutTransforms.expand(second, append(batch, inner, columns));
        output = result(batch, a.rank() == 1, b.rank() == 1);
    }

    TensorLayout left() { return left; }
    TensorLayout right() { return right; }
    TensorShape output() { return output; }
    int rows() { return rows; }
    int inner() { return inner; }
    int columns() { return columns; }

    private static TensorShape prefix(TensorShape shape) {
        return new TensorShape(Arrays.copyOf(shape.dimensions(), shape.rank() - 2));
    }

    private static int[] append(TensorShape batch, int rows, int columns) {
        int[] result = Arrays.copyOf(batch.dimensions(), batch.rank() + 2);
        result[batch.rank()] = rows;
        result[batch.rank() + 1] = columns;
        return result;
    }

    private TensorShape result(TensorShape batch, boolean leftVector, boolean rightVector) {
        int[] dimensions = Arrays.copyOf(batch.dimensions(), batch.rank() + 2);
        int rank = batch.rank();
        if (!leftVector) dimensions[rank++] = rows;
        if (!rightVector) dimensions[rank++] = columns;
        return new TensorShape(Arrays.copyOf(dimensions, rank));
    }
}
