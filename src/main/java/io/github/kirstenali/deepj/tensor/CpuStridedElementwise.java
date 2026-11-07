package io.github.kirstenali.deepj.tensor;

/** Visits each strided row once instead of dividing logical coordinates for every value. */
final class CpuStridedElementwise {
    private record Span(float[] values, TensorLayout layout, int step) {
        static Span of(Tensor tensor, TensorLayout layout) {
            return new Span(tensor.memory().floats(), layout, (int) layout.stride(-1));
        }
    }

    private CpuStridedElementwise() {}

    static boolean binary(Tensor left, Tensor right, Tensor output, CpuNdElementwise.Binary operation) {
        if (output.rank() == 0 || left.dtype() != DType.FLOAT32 || right.dtype() != DType.FLOAT32) return false;
        Span a = Span.of(left, LayoutTransforms.expand(left.layout(), output.shape()));
        Span b = Span.of(right, LayoutTransforms.expand(right.layout(), output.shape()));
        Span result = Span.of(output, output.layout());
        CpuDenseElementwise.ranges(output.numel(), (first, last) -> rows(a, b, result, operation, first, last));
        return true;
    }

    private static void rows(Span a, Span b, Span output, CpuNdElementwise.Binary operation, int first, int last) {
        int width = output.layout.shape().size(-1);
        while (first < last) {
            int count = Math.min(last - first, width - first % width);
            values(a, b, output, operation, first, count);
            first += count;
        }
    }

    private static void values(Span a, Span b, Span output, CpuNdElementwise.Binary operation, int first, int count) {
        int ai = a.layout.addressAt(first), bi = b.layout.addressAt(first), oi = output.layout.addressAt(first);
        for (int index = 0; index < count; index++) {
            output.values[oi] = operation.apply(a.values[ai], b.values[bi]);
            ai += a.step;
            bi += b.step;
            oi += output.step;
        }
    }
}
