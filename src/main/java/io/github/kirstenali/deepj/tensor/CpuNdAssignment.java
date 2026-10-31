package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Assigns logical values with explicit dtype conversion, including overlapping source/destination views. */
public final class CpuNdAssignment {
    private CpuNdAssignment() {}

    public static void apply(Tensor target, Tensor source) {
        Tensor.requireSameShape(target, source, "Assignment");
        target.layout().requireWritable();
        if (target == source) return;
        Tensor input = target.memory() == source.memory() ? new Tensor(source) : source;
        target.materialize();
        input.materialize();
        if (target.isContiguous() && input.isContiguous()) {
            dense(target, input);
            return;
        }
        for (int index = 0; index < target.numel(); index++) {
            int destination = target.layout().addressAt(index), origin = input.layout().addressAt(index);
            if (target.dtype() == input.dtype()) input.memory().copyTo(target.memory(), origin, destination, 1);
            else target.memory().set(destination, input.memory().get(origin));
        }
    }

    private static void dense(Tensor target, Tensor input) {
        if (target.dtype() == input.dtype()) {
            input.memory().copyTo(target.memory(), input.storageOffset(), target.storageOffset(), target.numel());
            return;
        }
        TensorMemory destination = target.memory(), origin = input.memory();
        int to = target.storageOffset(), from = input.storageOffset();
        DeepJExecutor.forRange(0, target.numel(), index -> destination.set(to + index, origin.get(from + index)));
    }
}
