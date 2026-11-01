package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;

/** Reset-after GRU, with reset/update/candidate gate order and separate projection biases. */
public final class GRUCell extends RecurrentCell {
    public GRUCell(int inputs, int hidden, Random random) {
        this(inputs, hidden, random, ParameterFactory.float32());
    }

    public GRUCell(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(inputs, hidden, 3, random, parameters);
    }

    @Override protected RecurrentState next(Tensor input, RecurrentState state) {
        Tensor projected = inputProjection.forward(input), previous = hiddenProjection.forward(state.hidden());
        Tensor reset = gate(projected, 0).add(gate(previous, 0)).sigmoidActivation();
        Tensor update = gate(projected, 1).add(gate(previous, 1)).sigmoidActivation();
        Tensor candidate = gate(projected, 2).add(reset.multiply(gate(previous, 2))).tanhActivation();
        Tensor hidden = candidate.add(update.multiply(state.hidden().subtract(candidate)));
        return new RecurrentState(hidden);
    }
}
