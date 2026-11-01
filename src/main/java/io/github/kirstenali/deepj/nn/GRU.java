package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import java.util.Random;

/** Batch-first GRU layer; compose layers with Sequential for stacked recurrence. */
public final class GRU extends Recurrent {
    public GRU(int inputs, int hidden, Random random) { super(new GRUCell(inputs, hidden, random)); }

    public GRU(int inputs, int hidden, Random random, ParameterFactory parameters) {
        super(new GRUCell(inputs, hidden, random, parameters));
    }
}
