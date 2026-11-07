package io.github.kirstenali.deepj.tensor;

/** Strided indexing resolves outer coordinates once per row and retains repeated-index addition order. */
final class CpuRowSelection {
    private record FloatRows(float[] values, TensorLayout layout, int step) {
        static FloatRows of(Tensor tensor) {
            return new FloatRows(tensor.memory().floats(), tensor.layout(), (int) tensor.layout().stride(-1));
        }
    }

    private CpuRowSelection() {}

    static boolean select(Tensor input, IndexSelection selection, Tensor output) {
        if (selection.axis() == input.rank() - 1 || input.layout().stride(-1) != 1) return false;
        if (output.numel() == 0) return true;
        int width = output.size(-1);
        for (int row = 0; row < output.numel() / width; row++) {
            int source = input.layout().addressAt(selection.sourceIndex(row * width));
            input.memory().copyTo(output.memory(), source, row * width, width);
        }
        return true;
    }

    static boolean add(Tensor target, IndexSelection selection, Tensor updates) {
        if (target.dtype() != DType.FLOAT32 || updates.dtype() != DType.FLOAT32) return false;
        if (updates.numel() == 0) return true;
        addRows(FloatRows.of(target), FloatRows.of(updates), selection, updates.size(-1));
        return true;
    }

    private static void addRows(FloatRows target, FloatRows updates, IndexSelection selection, int width) {
        boolean last = selection.axis() == target.layout.shape().rank() - 1;
        for (int row = 0; row < selection.output().elements() / width; row++) {
            int logical = last ? row * target.layout.shape().size(-1) : selection.sourceIndex(row * width);
            int destination = target.layout.addressAt(logical), source = updates.layout.addressAt(row * width);
            if (last) addSelected(target, updates, selection, destination, source, width);
            else addRow(target, updates, destination, source, width);
        }
    }

    private static void addSelected(FloatRows target, FloatRows updates, IndexSelection selection,
                                     int destination, int source, int width) {
        for (int column = 0; column < width; column++) {
            target.values[destination + selection.indexAt(column) * target.step]
                    += updates.values[source + column * updates.step];
        }
    }

    private static void addRow(FloatRows target, FloatRows updates, int destination, int source, int width) {
        for (int column = 0; column < width; column++) {
            target.values[destination + column * target.step] += updates.values[source + column * updates.step];
        }
    }
}
