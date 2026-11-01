package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import java.util.Random;

/** Batch-first LSTM layer; compose layers with Sequential for stacked recurrence. */
public final class LSTM extends Recurrent {
    public LSTM(int inputs, int hidden, Random random) { super(new LSTMCell(inputs, hidden, random)); }

    public LSTM(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(new LSTMCell(inputs, hidden, random, parameters));
    }
}
