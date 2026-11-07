package io.github.kirstenali.deepj.tensor;

/** Dense CPU additions use a direct vectorizable loop, with workers reserved for larger buffers. */
public final class CpuAddition {
    private static final int PARALLEL_MINIMUM = 262144;
    private static final int BLOCK = 32768;

    private CpuAddition() {}

    public static Tensor apply(Tensor left, Tensor right) {
        if (!CpuDenseElementwise.supports(left, right) && !CpuCompactPointwise.supports(left, right)
                && !CpuMatrixAddition.supports(left, right)) {
            return CpuNdElementwise.binary(left, right, Float::sum);
        }
        Tensor result = new Tensor(DType.promote(left.dtype(), right.dtype()), left.shape());
        into(left, right, result);
        return result;
    }

    public static void into(Tensor left, Tensor right, Tensor output) {
        if (CpuCompactPointwise.apply(left, right, output, CpuCompactPointwise.ADD)) return;
        if (CpuMatrixAddition.apply(left, right, output)) return;
        if (!CpuDenseElementwise.supports(left, right, output) || overlaps(left, output) || overlaps(right, output)) {
            CpuNdElementwise.binaryInto(left, right, output, Float::sum);
            return;
        }
        output.layout().requireWritable();
        left.materialize();
        right.materialize();
        output.materialize();
        add(CpuFloatSpan.of(left), CpuFloatSpan.of(right), CpuFloatSpan.of(output), output.numel());
    }

    private static boolean overlaps(Tensor input, Tensor output) {
        return input.memory() == output.memory() && input.storageOffset() != output.storageOffset();
    }

    private static void add(CpuFloatSpan left, CpuFloatSpan right, CpuFloatSpan output, int count) {
        if (count < PARALLEL_MINIMUM) {
            values(left, right, output, 0, count);
            return;
        }
        CpuRanges.balance((count - 1) / BLOCK + 1, block ->
                values(left, right, output, block * BLOCK, (int) Math.min(count, (long) (block + 1) * BLOCK)));
    }

    private static void values(CpuFloatSpan left, CpuFloatSpan right, CpuFloatSpan output, int first, int last) {
        float[] a = left.values(), b = right.values(), c = output.values();
        int ao = left.offset(), bo = right.offset(), co = output.offset();
        for (int index = first; index < last; index++) {
            c[co + index] = a[ao + index] + b[bo + index];
        }
    }
}
