package io.github.kirstenali.deepj.tensor;

/** Dense axis selection copies complete trailing blocks without per-element layout decoding. */
final class CpuContiguousSelection {
    private CpuContiguousSelection() {}

    static boolean apply(Tensor input, IndexSelection selection, Tensor output) {
        if (!input.isContiguous()) return false;
        if (output.numel() == 0) return true;
        int block = Math.toIntExact(input.layout().shape().denseStrides()[selection.axis()]);
        copyBlocks(input, selection, output, block);
        return true;
    }

    private static void copyBlocks(Tensor input, IndexSelection selection, Tensor output, int block) {
        int selected = selection.output().size(selection.axis()), width = input.size(selection.axis());
        for (int row = 0; row < output.numel() / block; row++) {
            int source = (row / selected * width + selection.indexAt(row % selected)) * block;
            input.memory().copyTo(output.memory(), input.storageOffset() + source, row * block, block);
        }
    }
}
