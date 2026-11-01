package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;

/** Elman recurrent cell with tanh activation. */
public final class RNNCell extends RecurrentCell {
    public RNNCell(int inputs, int hidden, Random random) {
        this(inputs, hidden, random, ParameterFactory.float32());
    }

    public RNNCell(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(inputs, hidden, 1, random, parameters);
    }

    @Override protected RecurrentState next(Tensor input, RecurrentState state) {
        Tensor hidden = inputProjection.forward(input).add(hiddenProjection.forward(state.hidden())).tanhActivation();
        return new RecurrentState(hidden);
    }
}
