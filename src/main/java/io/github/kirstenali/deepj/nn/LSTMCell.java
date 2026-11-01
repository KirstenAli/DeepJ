package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;

/** LSTM with input/forget/candidate/output gate order and explicit cell memory. */
public final class LSTMCell extends RecurrentCell {
    public LSTMCell(int inputs, int hidden, Random random) {
        this(inputs, hidden, random, ParameterFactory.float32());
    }

    public LSTMCell(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(inputs, hidden, 4, random, parameters);
    }

    @Override protected boolean hasCellMemory() { return true; }

    @Override protected RecurrentState next(Tensor input, RecurrentState state) {
        Tensor gates = inputProjection.forward(input).add(hiddenProjection.forward(state.hidden()));
        Tensor write = gate(gates, 0).sigmoidActivation().multiply(gate(gates, 2).tanhActivation());
        Tensor memory = gate(gates, 1).sigmoidActivation().multiply(state.cell()).add(write);
        Tensor hidden = gate(gates, 3).sigmoidActivation().multiply(memory.tanhActivation());
        return new RecurrentState(hidden, memory);
    }
}
