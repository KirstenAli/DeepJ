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
        batch(plan);
        return true;
    }

    private static boolean available(CpuProductPlan p) {
        return p.left.floats() != null && p.right.floats() != null && CpuBlas.AVAILABLE;
    }

    private static boolean compatible(CpuProductPlan p) {
        if (!compatibleLeft(p)) return false;
        return p.columnStride() == 1 && p.rightInnerStride >= p.shape.columns()
                || p.rightInnerStride == 1 && p.columnStride() >= p.shape.inner();
    }

    private static boolean compatibleLeft(CpuProductPlan p) {
        if (p.leftInnerStride == 1) return p.shape.rows() == 1 || p.leftRowStride >= p.shape.inner();
        return p.leftRowStride == 1 && p.leftInnerStride >= p.shape.rows();
    }

    private static void batch(CpuProductPlan p) {
        int m = p.shape.rows(), k = p.shape.inner(), n = p.shape.columns();
        int[] shape = new int[10 + 3 * p.batches()];
        int[] header = header(p);
        System.arraycopy(header, 0, shape, 0, header.length);
        for (int batch = 0; batch < p.batches(); batch++) {
            shape[10 + batch * 3] = p.shape.left().addressAt(batch * m * k);
            shape[11 + batch * 3] = p.shape.right().addressAt(batch * k * n);
            shape[12 + batch * 3] = batch * m * n;
        }
        CpuBlas.product(p.left.floats(), p.right.floats(), p.output.floats(), shape);
    }

    private static int leadingLeft(CpuProductPlan p) {
        if (p.leftInnerStride != 1) return p.leftInnerStride;
        return p.shape.rows() == 1 ? p.shape.inner() : p.leftRowStride;
    }

    private static int[] header(CpuProductPlan p) {
        boolean transpose = p.leftInnerStride != 1;
        int lda = leadingLeft(p);
        int ldb = p.columnStride() == 1 ? p.rightInnerStride : p.columnStride();
        int flags = (transpose ? 2 : 0) | (p.columnStride() == 1 ? 0 : 1);
        int serial = !DeepJExecutor.isParallelEnabled() || DeepJExecutor.getNumThreads() == 1 ? 1 : 0;
        return new int[]{p.shape.rows(), p.shape.columns(), p.shape.inner(), 0, 0, 0, lda, ldb, flags, serial};
    }
}
