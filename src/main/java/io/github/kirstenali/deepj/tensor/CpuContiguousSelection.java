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

    static boolean add(Tensor target, IndexSelection selection, Tensor updates) {
        if (target.dtype() != DType.FLOAT32 || updates.dtype() != DType.FLOAT32
                || !target.isContiguous() || !updates.isContiguous()) return false;
        if (updates.numel() == 0) return true;
        int block = Math.toIntExact(target.layout().shape().denseStrides()[selection.axis()]);
        int selected = selection.output().size(selection.axis()), width = target.size(selection.axis());
        for (int row = 0; row < updates.numel() / block; row++) {
            int destination = (row / selected * width + selection.indexAt(row % selected)) * block;
            addBlock(target, updates, destination, row * block, block);
        }
        return true;
    }

    private static void addBlock(Tensor target, Tensor updates, int destination, int source, int count) {
        float[] output = target.memory().floats(), input = updates.memory().floats();
        int start = target.storageOffset() + destination, first = updates.storageOffset() + source;
        for (int index = 0; index < count; index++) { output[start + index] += input[first + index]; }
    }
}
