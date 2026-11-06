package io.github.kirstenali.deepj.tensor;

/** Compact CPU matrix slots favor contiguous dot products; tall lookup tables stay row-major. */
public final class CpuMatrixStorage {
    private CpuMatrixStorage() {}

    public static Tensor zeros(DType dtype, int... shape) {
        if (!columnMajor(dtype, shape)) return new Tensor(dtype, shape);
        Tensor storage = new Tensor(dtype, shape[1], shape[0]);
        return TensorViews.permute(storage, 1, 0);
    }

    private static boolean columnMajor(DType dtype, int[] shape) {
        if (dtype == DType.FLOAT32 || shape.length != 2 || !CpuBlas.AVAILABLE) return false;
        long elements = (long) shape[0] * shape[1];
        return elements >= 262144 && shape[0] > 1 && shape[1] > 1 && shape[0] <= 8L * shape[1];
    }
}
