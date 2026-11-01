package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import java.util.Random;

/** Batch-first RNN layer; compose layers with Sequential for stacked recurrence. */
public final class RNN extends Recurrent {
    public RNN(int inputs, int hidden, Random random) { super(new RNNCell(inputs, hidden, random)); }

    public RNN(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(new RNNCell(inputs, hidden, random, parameters));
    }
}
