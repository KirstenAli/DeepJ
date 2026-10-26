package io.github.kirstenali.deepj.tensor;

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
        for (int index = 0; index < target.numel(); index++) {
            target.memory().set(target.storageOffset() + index, input.memory().get(input.storageOffset() + index));
        }
    }
}
