package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

final class RecurrentCell {
    record State(Tensor hidden, Tensor cell) {}

    private RecurrentCell() {}

    static State step(RecurrentInputs r, Tensor input, State previous, int direction) {
        Tensor projected = r.project(input, r.weights.select(0, direction), r.bias(direction, false));
        Tensor memory = r.project(previous.hidden(), r.recurrent.select(0, direction), r.bias(direction, true));
        return switch (r.context.operator()) {
            case "RNN" -> new State(r.activate(projected.add(memory), direction, 0), previous.cell());
            case "GRU" -> gru(r, projected, memory, previous, direction);
            default -> lstm(r, projected.add(memory), previous, direction);
        };
    }

    private static State gru(RecurrentInputs r, Tensor projected, Tensor memory, State previous, int direction) {
        Tensor update = r.activate(r.gate(projected, 0).add(r.gate(memory, 0)), direction, 0);
        Tensor reset = r.activate(r.gate(projected, 1).add(r.gate(memory, 1)), direction, 0);
        Tensor candidateMemory = r.context.integer("linear_before_reset", 0) == 1 ? reset.multiply(r.gate(memory, 2))
                        : r.project(reset.multiply(previous.hidden()), r.recurrent.select(0, direction)
                                        .slice(0, 2 * r.hidden, 3 * r.hidden), r.gate(r.bias(direction, true), 2));
        Tensor candidate = r.activate(r.gate(projected, 2).add(candidateMemory), direction, 1);
        return new State(update.neg().addScalar(1).multiply(candidate).add(update.multiply(previous.hidden())),
                previous.cell());
    }

    private static State lstm(RecurrentInputs r, Tensor projected, State previous, int direction) {
        Tensor peepholes = r.peepholes.select(0, direction);
        Tensor input = r.activate(r.gate(projected, 0).add(r.gate(peepholes, 0).multiply(previous.cell())), direction,
                        0);
        Tensor forget = r.context.integer("input_forget", 0) == 1 ? input.neg().addScalar(1) : r.activate(
                                r.gate(projected, 2).add(r.gate(peepholes, 2).multiply(previous.cell())), direction, 0);
        Tensor cell = forget.multiply(previous.cell())
                        .add(input.multiply(r.activate(r.gate(projected, 3), direction, 1)));
        Tensor output = r.activate(r.gate(projected, 1).add(r.gate(peepholes, 1).multiply(cell)), direction, 0);
        return new State(output.multiply(r.activate(cell, direction, 2)), cell);
    }
}
