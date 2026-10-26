package io.github.kirstenali.deepj.tensor;

/** Logical dimensions for the compatibility API; never inferred from a GPU allocation's physical rows. */
public record PackedMatmulShape(int leftRows, int rightRows, int rows, int cols, int inner,
                                boolean transposeLeft, boolean transposeRight) {
    public static PackedMatmulShape of(Tensor left, Tensor right, int batches, boolean ta, boolean tb) {
        requireHeads(left.matrixRows(), batches);
        requireHeads(right.matrixRows(), batches);
        int leftRows = left.matrixRows() / batches, rightRows = right.matrixRows() / batches;
        int rows = ta ? left.matrixCols() : leftRows, inner = ta ? leftRows : left.matrixCols();
        int rightInner = tb ? right.matrixCols() : rightRows, cols = tb ? rightRows : right.matrixCols();
        if (inner != rightInner) throw new IllegalArgumentException("Batched matmul shape mismatch");
        return new PackedMatmulShape(leftRows, rightRows, rows, cols, inner, ta, tb);
    }

    private static void requireHeads(int dimension, int heads) {
        if (heads < 1 || dimension % heads != 0) {
            throw new IllegalArgumentException("Dimension must be divisible by a positive head count");
        }
    }
}
