package io.github.kirstenali.deepj.tensor;

/** CPU-only products operate on typed storage and partition independent output tiles. */
public final class CpuMatmul {
    private CpuMatmul() {}

    public static Tensor attempt(Tensor left, Tensor right) { return attempt(left, right, null); }

    static Tensor attempt(Tensor left, Tensor right, int[] outputShape) {
        var shape = new NdMatmulShape(left, right);
        if (!eligible(shape, left, right)) return null;
        Tensor output = new Tensor(DType.promote(left.dtype(), right.dtype()), dimensions(shape, outputShape));
        if (output.numel() == 0 || shape.inner() == 0) return output;
        left.materialize();
        right.materialize();
        var plan = new CpuProductPlan(shape, left, right, output);
        if (CpuBlasProduct.apply(plan)) return output;
        if (CpuCompactProduct.apply(plan)) return output;
        if (shape.rows() > 4) return null;
        multiply(plan);
        return output;
    }

    private static int[] dimensions(NdMatmulShape shape, int[] requested) {
        return requested == null ? shape.output().dimensions() : requested;
    }

    private static boolean eligible(NdMatmulShape shape, Tensor left, Tensor right) {
        boolean dense = CpuNdElementwise.matrix(left) && CpuNdElementwise.matrix(right);
        long work = (long) shape.rows() * shape.columns() * shape.inner();
        if (!dense && shape.rows() <= 4) return true;
        return work >= 262144 && CpuBlas.AVAILABLE;
    }

    private static void multiply(CpuProductPlan plan) {
        long work = (long) plan.shape.rows() * plan.shape.columns() * plan.shape.inner();
        if (work < 32768) {
            for (int tile = 0; tile < plan.tiles(); tile++) { multiply(plan, tile); }
        } else CpuRanges.forRange(0, plan.tiles(), tile -> multiply(plan, tile));
    }

    private static void multiply(CpuProductPlan plan, int index) {
        CpuProductTile tile = plan.tile(index);
        if (plan.columnStride() == 1) CpuRowProduct.apply(tile);
        else CpuDotProduct.apply(tile);
    }
}
