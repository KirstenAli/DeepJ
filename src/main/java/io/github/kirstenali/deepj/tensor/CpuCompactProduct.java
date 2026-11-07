package io.github.kirstenali.deepj.tensor;

/** Compact weights are decoded in bounded CPU tiles, without keeping an expanded model copy. */
final class CpuCompactProduct {
    private static final int ROWS = 64;
    private CpuCompactProduct() {}

    static boolean apply(CpuProductPlan plan) {
        if (!CpuBlas.AVAILABLE) return false;
        if (plan.left.type() == 1 && plan.right.type() == 1 && plan.shape.rows() != 1) return false;
        if ((long) plan.shape.rows() * plan.shape.inner() * plan.shape.columns() < minimumWork(plan)) return false;
        int width = width(plan);
        CpuRanges.balance(plan.tiles(ROWS, width), index -> tile(plan.tile(index, ROWS, width)));
        return true;
    }

    private static int minimumWork(CpuProductPlan plan) {
        boolean compactBatch = plan.batches() > 1 && (plan.left.type() != 1 || plan.right.type() != 1);
        return compactBatch ? 16384 : 262144;
    }

    private static int width(CpuProductPlan p) {
        if (p.shape.rows() != 1) return p.shape.rows() <= ROWS && p.shape.inner() >= 2048 ? 128 : 512;
        return p.columnStride() != 1 && p.shape.columns() <= 16384 ? 64 : 256;
    }

    private static void tile(CpuProductTile tile) {
        var p = tile.plan();
        Object output = p.output.dtype() == DType.FLOAT32 ? p.output.floats() : p.output.shorts();
        CpuBlas.compactProduct(p.left.storage(), p.right.storage(), output,
                new int[]{tile.rows(), tile.columns(), p.shape.inner(), tile.left(), tile.right(), tile.output(),
                    p.leftRowStride, p.leftInnerStride, p.rightInnerStride, p.columnStride(), p.shape.columns(),
                    p.left.type(), p.right.type(), p.output.dtype().storageCode()});
    }
}
