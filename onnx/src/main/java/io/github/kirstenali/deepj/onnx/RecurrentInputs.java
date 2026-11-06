package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.List;

final class RecurrentInputs {
    final OnnxContext context;
    final Tensor input, weights, recurrent, bias, initial, cell, peepholes;
    final int[] lengths;
    final int hidden, directions, layout, gates;

    RecurrentInputs(OnnxContext context) {
        this.context = context;
        layout = layout(context);
        input = layout == 0 ? context.tensor(0) : context.tensor(0).transpose(0, 1);
        weights = context.tensor(1);
        recurrent = context.tensor(2);
        hidden = context.integer("hidden_size", recurrent.size(2));
        directions = context.string("direction", "forward").equals("bidirectional") ? 2 : 1;
        gates = gateCount(context.operator());
        validate();
        bias = optional(3, directions, 2 * gates * hidden);
        initial = state(5);
        cell = gates == 4 ? state(6) : zeros(directions, input.size(1), hidden);
        peepholes = optional(7, directions, 3 * hidden);
        lengths = sequenceLengths();
        validateStates();
    }

    private Tensor zeros(int... shape) {
        return Tensor.backend().zeros(input.dtype(), shape);
    }

    private Tensor state(int index) {
        if (!context.hasInput(index)) return zeros(directions, input.size(1), hidden);
        Tensor value = context.tensor(index);
        return layout == 0 ? value : value.transpose(0, 1);
    }

    private void validate() {
        if (!List.of("forward", "reverse", "bidirectional")
                .contains(context.string("direction", "forward"))) {
            throw new IllegalArgumentException("Unsupported RNN direction");
        }
        if (input.rank() != 3
                || !weights.hasShape(directions, gates * hidden, input.size(2))
                || !recurrent.hasShape(directions, gates * hidden, hidden)) {
            throw new IllegalArgumentException("RNN weight or input shape differs");
        }
    }

    private int[] sequenceLengths() {
        int[] values = new int[input.size(1)];
        Arrays.fill(values, input.size(0));
        if (context.hasInput(4)) values = context.input(4).ints();
        if (values.length != input.size(1))
            throw new IllegalArgumentException("RNN sequence-length batch differs");
        for (int value : values) {
            if (value < 0 || value > input.size(0))
                throw new IllegalArgumentException("Invalid RNN sequence length");
        }
        return values;
    }

    boolean reverse(int direction) {
        return direction == 1 || context.string("direction", "forward").equals("reverse");
    }

    Tensor gate(Tensor tensor, int index) {
        return tensor.slice(-1, index * hidden, (index + 1) * hidden);
    }

    Tensor bias(int direction, boolean recurrentPart) {
        int start = recurrentPart ? gates * hidden : 0;
        return bias.select(0, direction).slice(0, start, start + gates * hidden);
    }

    Tensor project(Tensor values, Tensor weight, Tensor bias) {
        return values.matmul(weight.transpose()).add(bias);
    }

    Tensor activate(Tensor input, int direction, int index) {
        List<String> defaults =
                gates == 4
                        ? List.of("Sigmoid", "Tanh", "Tanh")
                        : gates == 3 ? List.of("Sigmoid", "Tanh") : List.of("Tanh");
        int slot = direction * defaults.size() + index;
        List<String> names = context.strings("activations");
        String name = names.isEmpty() ? defaults.get(index) : names.get(slot);
        Tensor value =
                context.hasAttribute("clip")
                        ? input.clamp(-context.floating("clip", 0), context.floating("clip", 0))
                        : input;
        return RecurrentActivation.apply(
                name,
                value,
                attribute("activation_alpha", slot),
                attribute("activation_beta", slot));
    }

    private Float attribute(String name, int index) {
        var attribute = context.attributes.get(name);
        return attribute == null || attribute.getFloatsCount() <= index
                ? null
                : attribute.getFloats(index);
    }

    Tensor masked(Tensor selected, Tensor fallback, int time) {
        boolean[] mask = new boolean[lengths.length];
        boolean all = true;
        for (int i = 0; i < mask.length; i++) {
            mask[i] = time < lengths[i];
            all &= mask[i];
        }
        return all
                ? selected
                : BooleanOperators.where(
                        OnnxValue.booleans(mask, mask.length, 1), selected, fallback);
    }

    private static int layout(OnnxContext context) {
        int layout = context.integer("layout", 0);
        if (layout != 0 && layout != 1)
            throw new IllegalArgumentException("RNN layout must be zero or one");
        return layout;
    }

    private static int gateCount(String operator) {
        return switch (operator) {
            case "LSTM" -> 4;
            case "GRU" -> 3;
            default -> 1;
        };
    }

    private Tensor optional(int position, int... shape) {
        return context.hasInput(position) ? context.tensor(position) : zeros(shape);
    }

    private void validateStates() {
        if (hidden < 1
                || !initial.hasShape(directions, input.size(1), hidden)
                || !cell.hasShape(directions, input.size(1), hidden)) {
            throw new IllegalArgumentException("Invalid RNN hidden or cell state shape");
        }
        if (!bias.hasShape(directions, 2 * gates * hidden)
                || !peepholes.hasShape(directions, 3 * hidden)) {
            throw new IllegalArgumentException("Invalid RNN bias or peephole shape");
        }
    }
}
