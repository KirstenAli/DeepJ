package io.github.kirstenali.deepj.tensor;

/** Validated ND broadcasting and physical strides are shared by the CPU tile kernels. */
final class CpuProductPlan {
    private static final int ROWS = 1;
    private static final int COLUMNS = 128;
    final NdMatmulShape shape;
    final CpuProductValues left, right;
    final TensorMemory output;
    final int leftRowStride, leftInnerStride, rightInnerStride;
    private final int columnStride, batches;

    CpuProductPlan(NdMatmulShape shape, Tensor left, Tensor right, Tensor output) {
        this.shape = shape;
        this.left = new CpuProductValues(left);
        this.right = new CpuProductValues(right);
        this.output = output.memory();
        leftRowStride = stride(shape.left(), -2);
        leftInnerStride = stride(shape.left(), -1);
        rightInnerStride = stride(shape.right(), -2);
        columnStride = stride(shape.right(), -1);
        batches = output.numel() / shape.rows() / shape.columns();
    }

    private static int stride(TensorLayout layout, int axis) {
        return layout.shape().size(axis) <= 1 ? 0 : Math.toIntExact(layout.stride(axis));
    }

    int columnStride() { return columnStride; }
    int batches() { return batches; }
    int tiles() { return tiles(ROWS, COLUMNS); }
    int tiles(int height, int width) {
        return batches * ((shape.rows() - 1) / height + 1) * ((shape.columns() - 1) / width + 1);
    }

    CpuProductTile tile(int index) { return tile(index, ROWS, COLUMNS); }

    CpuProductTile tile(int index, int height, int width) {
        int columnTiles = (shape.columns() - 1) / width + 1, rowTiles = (shape.rows() - 1) / height + 1;
        int batch = index / columnTiles / rowTiles;
        int row = index / columnTiles % rowTiles * height, column = index % columnTiles * width;
        int count = Math.min(height, shape.rows() - row), columns = Math.min(width, shape.columns() - column);
        int a = shape.left().addressAt((batch * shape.rows() + row) * shape.inner());
        int b = shape.right().addressAt(batch * shape.inner() * shape.columns() + column);
        int result = (batch * shape.rows() + row) * shape.columns() + column;
        return new CpuProductTile(this, count, columns, a, b, result);
    }
}
