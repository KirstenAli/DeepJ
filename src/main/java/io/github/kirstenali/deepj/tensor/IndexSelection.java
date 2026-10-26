package io.github.kirstenali.deepj.tensor;

/** A one-dimensional index list selects one axis without flattening the other dimensions. */
final class IndexSelection {
    private final TensorShape input, output;
    private final int axis;
    private final int[] indices;
    private final long trailing;

    IndexSelection(TensorShape input, int axis, int[] indices) {
        this.input = input;
        this.axis = input.axis(axis);
        this.indices = indices.clone();
        for (int index : this.indices) {
            if (index < 0 || index >= input.size(this.axis)) throw new IndexOutOfBoundsException(index);
        }
        int[] dimensions = input.dimensions();
        dimensions[this.axis] = indices.length;
        output = new TensorShape(dimensions);
        trailing = input.denseStrides()[this.axis];
    }

    TensorShape output() { return output; }
    int axis() { return axis; }

    int sourceIndex(int outputIndex) {
        long row = outputIndex / trailing;
        long selected = indices[(int) (row % indices.length)];
        return Math.toIntExact((row / indices.length * input.size(axis) + selected) * trailing
                + outputIndex % trailing);
    }

    static IndexSelection requireUpdates(Tensor target, Tensor updates, int axis, int[] indices) {
        var selection = new IndexSelection(target.layout().shape(), axis, indices);
        if (!selection.output().equals(updates.layout().shape())) {
            throw new IllegalArgumentException("Indexed updates must match the selected tensor shape");
        }
        if (DType.promote(target.dtype(), updates.dtype()) != target.dtype()) {
            throw new IllegalArgumentException("Indexed updates cannot implicitly narrow dtype");
        }
        target.layout().requireWritable();
        return selection;
    }
}
