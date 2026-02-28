package org.DeepJ.ann.transformer;

import org.DeepJ.ann.activations.ActivationFunction;
import org.DeepJ.ann.activations.GELU;
import org.DeepJ.ann.layers.transformer.TransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Convenience builder for assembling transformer stacks.
 *
 * <p>DeepJ is transformer-oriented: this builder replaces generic network builders.
 */
public final class TransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private Supplier<ActivationFunction> ffnActivationFactory = GELU::new;
    private long seed = 42;

    public TransformerBuilder dModel(int dModel) {
        this.dModel = dModel;
        return this;
    }

    public TransformerBuilder nHeads(int nHeads) {
        this.nHeads = nHeads;
        return this;
    }

    public TransformerBuilder dFF(int dFF) {
        this.dFF = dFF;
        return this;
    }

    public TransformerBuilder nLayers(int nLayers) {
        this.nLayers = nLayers;
        return this;
    }

    /** Activation used inside the FFN. Default: GELU. */
    public TransformerBuilder ffnActivation(Supplier<ActivationFunction> activationFactory) {
        if (activationFactory == null) throw new IllegalArgumentException("activationFactory must not be null");
        this.ffnActivationFactory = activationFactory;
        return this;
    }

    public TransformerBuilder seed(long seed) {
        this.seed = seed;
        return this;
    }

    public TransformerStack build() {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (dFF <= 0) throw new IllegalArgumentException("dFF must be > 0");
        if (nLayers <= 0) throw new IllegalArgumentException("nLayers must be > 0");

        Random rnd = new Random(seed);
        List<TransformerBlock> blocks = new ArrayList<>(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.add(new TransformerBlock(dModel, nHeads, dFF, ffnActivationFactory.get(), rnd));
        }
        return new TransformerStack(blocks);
    }
}
