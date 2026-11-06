package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Large single-precision CPU products use BLAS directly on their physical storage. */
final class CpuBlasProduct {
    private CpuBlasProduct() {}

    static boolean apply(CpuProductPlan plan) {
        if (plan.shape.rows() == 1 && plan.columnStride() != 1) return false;
        if (!available(plan)) return false;
        if (!compatible(plan) || (long) plan.shape.rows() * plan.shape.inner() * plan.shape.columns() < 262144) {
            return false;
        }
        for (int batch = 0; batch < plan.batches(); batch++) { batch(plan, batch); }
        return true;
    }

    private static boolean available(CpuProductPlan p) {
        return p.left.floats() != null && p.right.floats() != null && CpuBlas.AVAILABLE;
    }

    private static boolean compatible(CpuProductPlan p) {
        if (p.leftInnerStride != 1 || p.leftRowStride < p.shape.inner() && p.shape.rows() > 1) return false;
        return p.columnStride() == 1 && p.rightInnerStride >= p.shape.columns()
                || p.rightInnerStride == 1 && p.columnStride() >= p.shape.inner();
    }

    private static void batch(CpuProductPlan p, int batch) {
        int m = p.shape.rows(), k = p.shape.inner(), n = p.shape.columns();
        int a = p.shape.left().addressAt(batch * m * k), b = p.shape.right().addressAt(batch * k * n);
        int lda = m == 1 ? k : p.leftRowStride, ldb = p.columnStride() == 1 ? p.rightInnerStride : p.columnStride();
        CpuBlas.product(p.left.floats(), p.right.floats(), p.output.floats(),
                new int[]{m, n, k, a, b, batch * m * n, lda, ldb, p.columnStride() == 1 ? 0 : 1,
                    !DeepJExecutor.isParallelEnabled() || DeepJExecutor.getNumThreads() == 1 ? 1 : 0});
    }
}
