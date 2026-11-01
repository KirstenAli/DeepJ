package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Batch-first recurrent sequence execution, with optional lengths and caller-owned initial/final state. */
public class Recurrent extends Module {
    private final RecurrentCell cell;

    public Recurrent(RecurrentCell cell) { this.cell = Objects.requireNonNull(cell, "cell"); }
    public final RecurrentCell cell() { return cell; }
    @Override protected List<Layer> children() { return List.of(cell); }
    @Override public final Tensor forward(Tensor input) { return run(input, null, null).sequence(); }

    public final RecurrentResult run(Tensor input, RecurrentState initial, int[] lengths) {
        validate(input, lengths);
        int[] saved = lengths == null ? null : lengths.clone();
        try (var scope = TensorScope.open()) {
            RecurrentResult result = unroll(input, initial, saved);
            scope.escape(result.sequence());
            scope.escape(result.state().hidden());
            if (result.state().cell() != null) scope.escape(result.state().cell());
            return result;
        }
    }

    private RecurrentResult unroll(Tensor input, RecurrentState initial, int[] lengths) {
        RecurrentState state = initial == null ? cell.initialState(input.size(0), input.dtype()) : initial;
        Tensor[] outputs = new Tensor[input.size(1)];
        for (int time = 0; time < outputs.length; time++) {
            RecurrentState candidate = cell.step(input.select(1, time), state);
            Tensor mask = lengths == null ? null : activeMask(input, lengths, time);
            state = mask == null ? candidate : blend(state, candidate, mask);
            outputs[time] = mask == null ? state.hidden() : state.hidden().multiply(mask);
        }
        return new RecurrentResult(TensorJoin.stack(1, outputs), state);
    }

    private void validate(Tensor input, int[] lengths) {
        NeuralChecks.rank(input, 3);
        NeuralChecks.width(input, cell.inputSize());
        if (lengths == null) return;
        if (lengths.length != input.size(0)) {
            throw new IllegalArgumentException("One sequence length per sample required");
        }
        if (Arrays.stream(lengths).anyMatch(length -> length < 0 || length > input.size(1))) {
            throw new IllegalArgumentException("Sequence length must be between zero and the time dimension");
        }
    }

    private static Tensor activeMask(Tensor input, int[] lengths, int time) {
        float[] values = new float[lengths.length];
        for (int index = 0; index < values.length; index++) { values[index] = time < lengths[index] ? 1 : 0; }
        return Tensor.from(values, input.dtype(), values.length, 1);
    }

    private static RecurrentState blend(RecurrentState old, RecurrentState next, Tensor mask) {
        Tensor hidden = interpolate(old.hidden(), next.hidden(), mask);
        Tensor memory = old.cell() == null ? null : interpolate(old.cell(), next.cell(), mask);
        return new RecurrentState(hidden, memory);
    }

    private static Tensor interpolate(Tensor old, Tensor next, Tensor mask) {
        return old.add(next.subtract(old).multiply(mask));
    }
}
