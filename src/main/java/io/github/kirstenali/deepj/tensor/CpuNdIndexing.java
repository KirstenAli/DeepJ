package io.github.kirstenali.deepj.tensor;

import java.util.LinkedHashMap;
import java.util.Map;

/** Logical ND index selection and deterministic repeated-index accumulation. */
public final class CpuNdIndexing {
    private CpuNdIndexing() {}

    public static Tensor select(Tensor input, int axis, int[] indices) {
        var selection = new IndexSelection(input.layout().shape(), axis, indices);
        if (selection.axis() == 0 && CpuNdElementwise.matrix(input)) {
            return TensorBackendDefaults.sliceRows(input, indices);
        }
        input.materialize();
        Tensor result = new Tensor(input.dtype(), selection.output().dimensions());
        for (int index = 0; index < result.numel(); index++) {
            int source = input.layout().addressAt(selection.sourceIndex(index));
            input.memory().copyTo(result.memory(), source, index, 1);
        }
        return result;
    }

    public static void add(Tensor target, int axis, int[] indices, Tensor updates) {
        IndexSelection selection = IndexSelection.requireUpdates(target, updates, axis, indices);
        Tensor source = target.memory() == updates.memory() ? new Tensor(updates) : updates;
        target.materialize();
        source.materialize();
        if (selection.axis() == 0 && CpuNdElementwise.matrix(target) && CpuNdElementwise.matrix(source)) {
            addMatrixRows(target, indices, source);
            return;
        }
        if (target.dtype() == DType.FLOAT32) addFloat32(target, source, selection);
        else addCompact(target, source, selection);
    }

    public static void addMatrixRows(Tensor target, int[] indices, Tensor updates) {
        float[] output = target.rawData(), input = updates.rawData();
        int columns = target.matrixCols();
        for (int row = 0; row < indices.length; row++) {
            int destination = indices[row] * columns, source = row * columns;
            for (int column = 0; column < columns; column++) {
                output[destination + column] += input[source + column];
            }
        }
    }

    private static void addFloat32(Tensor target, Tensor updates, IndexSelection selection) {
        for (int index = 0; index < updates.numel(); index++) {
            int destination = target.layout().addressAt(selection.sourceIndex(index));
            float value = updates.memory().get(updates.layout().addressAt(index));
            target.memory().set(destination, target.memory().get(destination) + value);
        }
    }

    private static void addCompact(Tensor target, Tensor updates, IndexSelection selection) {
        Map<Integer, Float> sums = new LinkedHashMap<>();
        for (int index = 0; index < updates.numel(); index++) {
            int destination = target.layout().addressAt(selection.sourceIndex(index));
            float value = updates.memory().get(updates.layout().addressAt(index));
            sums.put(destination, sums.getOrDefault(destination, target.memory().get(destination)) + value);
        }
        sums.forEach((address, value) -> target.memory().set(address, value));
    }

    public static Tensor gradient(Tensor gradient, int[] shape, int axis, int[] indices) {
        Tensor result = new Tensor(DType.FLOAT32, shape);
        add(result, axis, indices, gradient);
        return result;
    }
}
