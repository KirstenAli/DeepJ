package io.github.kirstenali.deepj.tensor;

/** Contiguous compact activations use bounded native tiles; no expanded parameter copy is retained. */
final class CpuCompactPointwise {
    static final int ADD = 0;
    static final int SWIGLU = 1;

    private CpuCompactPointwise() {}

    static boolean supports(Tensor left, Tensor right) {
        return (left.dtype() != DType.FLOAT32 || right.dtype() != DType.FLOAT32)
                && CpuBlas.AVAILABLE && left.numel() >= 256 && left.hasSameShape(right)
                && left.isContiguous() && right.isContiguous();
    }

    static boolean apply(Tensor left, Tensor right, Tensor output, int operation) {
        if (!supports(left, right) || !output.hasSameShape(left) || !output.isContiguous()) return false;
        if (output.dtype() != DType.promote(left.dtype(), right.dtype()) || overlap(left, output)
                || overlap(right, output)) return false;
        output.layout().requireWritable();
        left.materialize();
        right.materialize();
        output.materialize();
        var a = new CpuProductValues(left);
        var b = new CpuProductValues(right);
        var c = new CpuProductValues(output);
        CpuDenseElementwise.ranges(output.numel(), (first, last) ->
                tile(left, right, output, a, b, c, first, last, operation));
        return true;
    }

    private static void tile(Tensor left, Tensor right, Tensor output, CpuProductValues a, CpuProductValues b,
                              CpuProductValues c, int first, int last, int operation) {
        int[] shape = {left.storageOffset() + first, right.storageOffset() + first, output.storageOffset() + first,
            last - first, a.type(), b.type(), c.type(), operation};
        if (CpuBlas.compactPointwise(a.storage(), b.storage(), c.storage(), shape)) return;
        for (int index = first; index < last; index++) {
            float value = a.get(left.storageOffset() + index), other = b.get(right.storageOffset() + index);
            float result = operation == ADD ? value + other : value * (1 / (1 + (float) Math.exp(-value))) * other;
            output.memory().set(output.storageOffset() + index, result);
        }
    }

    private static boolean overlap(Tensor input, Tensor output) {
        return input.memory() == output.memory() && input.storageOffset() != output.storageOffset();
    }
}
