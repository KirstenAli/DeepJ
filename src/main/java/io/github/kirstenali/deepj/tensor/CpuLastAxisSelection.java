package io.github.kirstenali.deepj.tensor;

/** Last-axis gathers resolve outer strides once per row and copy the original storage bits. */
final class CpuLastAxisSelection {
    @FunctionalInterface private interface Row { void copy(int source, int target); }

    private CpuLastAxisSelection() {}

    static boolean apply(Tensor input, IndexSelection selection, Tensor output) {
        if (selection.axis() != input.rank() - 1) return false;
        if (output.numel() == 0) return true;
        int columns = output.size(-1), rows = output.numel() / columns;
        Row copy = copier(input, selection, output);
        if (output.numel() < 8192) copyRows(input, copy, columns, 0, rows);
        else CpuRanges.balance((rows - 1) / 16 + 1, block ->
                copyRows(input, copy, columns, block * 16, Math.min(rows, (block + 1) * 16)));
        return true;
    }

    private static void copyRows(Tensor input, Row copy, int columns, int first, int last) {
        int width = input.size(-1);
        for (int row = first; row < last; row++) {
            copy.copy(input.layout().addressAt(row * width), row * columns);
        }
    }

    private static Row copier(Tensor input, IndexSelection selection, Tensor output) {
        long stride = input.layout().stride(-1);
        if (input.dtype() == DType.FLOAT32) {
            return floats(input.memory().floats(), output.memory().floats(), selection, stride);
        }
        return shorts(input.memory().shorts(), output.memory().shorts(), selection, stride);
    }

    private static Row floats(float[] input, float[] output, IndexSelection selection, long stride) {
        int width = selection.output().size(-1);
        return (source, target) -> {
            for (int column = 0; column < width; column++) {
                output[target + column] = input[(int) (source + selection.indexAt(column) * stride)];
            }
        };
    }

    private static Row shorts(short[] input, short[] output, IndexSelection selection, long stride) {
        int width = selection.output().size(-1);
        return (source, target) -> {
            for (int column = 0; column < width; column++) {
                output[target + column] = input[(int) (source + selection.indexAt(column) * stride)];
            }
        };
    }
}
