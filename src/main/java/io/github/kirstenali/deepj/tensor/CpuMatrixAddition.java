package io.github.kirstenali.deepj.tensor;

/** Adds transposed FLOAT32 matrices in small contiguous tiles, including gradient accumulation. */
final class CpuMatrixAddition {
    private CpuMatrixAddition() {}

    static boolean supports(Tensor left, Tensor right) {
        return CpuBlas.AVAILABLE && left.numel() >= 4096 && matrix(left) && matrix(right)
                && left.hasSameShape(right) && (!left.isContiguous() || !right.isContiguous());
    }

    static boolean apply(Tensor left, Tensor right, Tensor output) {
        if (!supports(left, right) || !denseOutput(left, output)) return false;
        output.layout().requireWritable();
        left = CpuNdElementwise.readable(left, output);
        right = CpuNdElementwise.readable(right, output);
        left.materialize();
        right.materialize();
        output.materialize();
        return CpuBlas.matrixAdd(left.memory().floats(), right.memory().floats(), output.memory().floats(),
                shape(left, right, output));
    }

    private static boolean denseOutput(Tensor input, Tensor output) {
        return output.dtype() == DType.FLOAT32 && output.isContiguous() && input.hasSameShape(output);
    }

    private static boolean matrix(Tensor input) {
        return input.rank() == 2 && input.dtype() == DType.FLOAT32
                && (input.layout().stride(0) == 1 || input.layout().stride(1) == 1);
    }

    private static int[] shape(Tensor left, Tensor right, Tensor output) {
        return new int[]{left.storageOffset(), right.storageOffset(), output.storageOffset(),
                output.matrixRows(), output.matrixCols(), (int) left.layout().stride(0),
                (int) left.layout().stride(1), (int) right.layout().stride(0), (int) right.layout().stride(1)};
    }
}
