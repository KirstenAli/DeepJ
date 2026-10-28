package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Pointwise kernels over contiguous storage of any rank, without per-value layout division. */
final class CpuDenseElementwise {
    private static final int BLOCK = 4096;
    @FunctionalInterface interface Range { void apply(int first, int last); }

    private CpuDenseElementwise() {}

    static boolean supports(Tensor left, Tensor right) {
        return dense(left) && dense(right) && left.hasSameShape(right);
    }

    static boolean supports(Tensor left, Tensor right, Tensor output) {
        return supports(left, right) && supports(left, output);
    }

    private static boolean dense(Tensor input) {
        return input.dtype() == DType.FLOAT32 && input.isContiguous();
    }

    static void unary(Tensor input, Tensor output, CpuNdElementwise.Unary operation) {
        var source = CpuFloatSpan.of(input);
        var result = CpuFloatSpan.of(output);
        ranges(output.numel(), (first, last) -> unaryValues(source, result, operation, first, last));
    }

    private static void unaryValues(CpuFloatSpan source, CpuFloatSpan output, CpuNdElementwise.Unary operation,
                                    int first, int last) {
        for (int index = first; index < last; index++) {
            output.set(index, operation.apply(source.get(index)));
        }
    }

    static void binary(Tensor left, Tensor right, Tensor output, CpuNdElementwise.Binary operation) {
        var a = CpuFloatSpan.of(left);
        var b = CpuFloatSpan.of(right);
        var result = CpuFloatSpan.of(output);
        ranges(output.numel(), (first, last) -> binaryValues(a, b, result, operation, first, last));
    }

    private static void binaryValues(CpuFloatSpan left, CpuFloatSpan right, CpuFloatSpan output,
                                     CpuNdElementwise.Binary operation, int first, int last) {
        for (int index = first; index < last; index++) {
            output.set(index, operation.apply(left.get(index), right.get(index)));
        }
    }

    static void ranges(int count, Range operation) {
        if (count == 0) return;
        DeepJExecutor.forRange(0, (count - 1) / BLOCK + 1, block ->
                operation.apply(block * BLOCK, (int) Math.min(count, (long) (block + 1) * BLOCK)));
    }
}
