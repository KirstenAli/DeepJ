package io.github.kirstenali.deepj.tensor;

/** Loaded CPU projections favor contiguous columns; tall lookup tables stay row-major. */
public final class CpuMatrixStorage {
    private CpuMatrixStorage() {}

    public static Tensor zeros(DType dtype, int... shape) {
        return dtype == DType.FLOAT32 ? new Tensor(dtype, shape) : zerosForLoading(dtype, shape);
    }

    public static Tensor zerosForLoading(DType dtype, int... shape) {
        if (!columnMajor(shape)) return new Tensor(dtype, shape);
        Tensor storage = new Tensor(dtype, shape[1], shape[0]);
        return TensorViews.permute(storage, 1, 0);
    }

    private static boolean columnMajor(int[] shape) {
        if (shape.length != 2 || !CpuBlas.AVAILABLE) return false;
        long elements = (long) shape[0] * shape[1];
        return elements >= 262144 && shape[0] > 1 && shape[1] > 1 && shape[0] <= 8L * shape[1];
    }
}
