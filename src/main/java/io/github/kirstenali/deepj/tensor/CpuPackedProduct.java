package io.github.kirstenali.deepj.tensor;

/** Packed CPU products allocate their public output shape directly, retaining sole storage ownership. */
public final class CpuPackedProduct {
    private CpuPackedProduct() {}

    public static Tensor attempt(Tensor left, Tensor right, int batches,
                                  boolean transposeLeft, boolean transposeRight) {
        var shape = PackedMatmulShape.of(left, right, batches, transposeLeft, transposeRight);
        try (var scope = TensorScope.open()) {
            Tensor a = operand(left, batches, transposeLeft);
            Tensor b = operand(right, batches, transposeRight);
            Tensor result = CpuMatmul.attempt(a, b, new int[]{batches * shape.rows(), shape.cols()});
            return result == null ? null : scope.escape(result);
        }
    }

    private static Tensor operand(Tensor input, int batches, boolean transpose) {
        TensorLayout source = input.layout();
        int rows = input.matrixRows() / batches;
        long batchStride = input.numel() == 0 ? 0 : Math.multiplyExact(source.stride(0), rows);
        var dimensions = new TensorShape(batches, rows, input.matrixCols());
        long[] strides = {batchStride, source.stride(0), source.stride(1)};
        TensorLayout layout = source.with(dimensions, strides, source.offset());
        return Tensor.aliasStorage(input, transpose ? LayoutTransforms.permute(layout, 0, 2, 1) : layout);
    }
}
