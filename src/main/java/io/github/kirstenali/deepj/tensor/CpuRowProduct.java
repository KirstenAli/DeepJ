package io.github.kirstenali.deepj.tensor;

/** Portable row-major products keep each sum in its original K order. */
final class CpuRowProduct {
    private CpuRowProduct() {}

    static void apply(CpuProductTile tile) { oneRow(tile, 0); }

    private static void oneRow(CpuProductTile tile, int row) {
        var plan = tile.plan();
        float[] sums = new float[tile.columns()];
        int left = tile.leftRow(row);
        for (int k = 0; k < plan.shape.inner(); k++) {
            float value = plan.left.get(left + k * plan.leftInnerStride);
            accumulate(sums, plan.right, tile.right() + k * plan.rightInnerStride, value);
        }
        tile.store(row, sums);
    }

    private static void accumulate(float[] sums, CpuProductValues right, int first, float value) {
        float[] values = right.floats();
        if (values != null) {
            for (int column = 0; column < sums.length; column++) { sums[column] += value * values[first + column]; }
            return;
        }
        for (int column = 0; column < sums.length; column++) {
            sums[column] += value * right.get(first + column);
        }
    }
}
