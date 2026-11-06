package io.github.kirstenali.deepj.tensor;

/** Transposed weights stay in place; independent dot products hide accumulation latency. */
final class CpuDotProduct {
    private CpuDotProduct() {}

    static void apply(CpuProductTile tile) {
        for (int row = 0; row < tile.rows(); row++) { row(tile, row); }
    }

    private static void row(CpuProductTile tile, int row) {
        int column = 0;
        for (; column + 4 <= tile.columns(); column += 4) { four(tile, row, column); }
        for (; column < tile.columns(); column++) { one(tile, row, column); }
    }

    private static void four(CpuProductTile tile, int row, int column) {
        var plan = tile.plan();
        int first = tile.right() + column * plan.columnStride(), step = plan.columnStride();
        float a = 0, b = 0, c = 0, d = 0;
        for (int k = 0; k < plan.shape.inner(); k++) {
            float value = plan.left.get(tile.leftRow(row) + k * plan.leftInnerStride);
            int right = first + k * plan.rightInnerStride;
            a += value * plan.right.get(right);
            b += value * plan.right.get(right + step);
            c += value * plan.right.get(right + 2 * step);
            d += value * plan.right.get(right + 3 * step);
        }
        storeFour(plan.output, tile.outputRow(row) + column, a, b, c, d);
    }

    private static void storeFour(TensorMemory output, int first, float a, float b, float c, float d) {
        output.set(first, a);
        output.set(first + 1, b);
        output.set(first + 2, c);
        output.set(first + 3, d);
    }

    private static void one(CpuProductTile tile, int row, int column) {
        var plan = tile.plan();
        int left = tile.leftRow(row), right = tile.right() + column * plan.columnStride();
        float sum = 0;
        for (int k = 0; k < plan.shape.inner(); k++) {
            sum += plan.left.get(left + k * plan.leftInnerStride) * plan.right.get(right + k * plan.rightInnerStride);
        }
        plan.output.set(tile.outputRow(row) + column, sum);
    }
}
