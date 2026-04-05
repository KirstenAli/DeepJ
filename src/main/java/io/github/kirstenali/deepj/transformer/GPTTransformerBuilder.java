package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.GPTTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Builder for GPT-style transformer stacks.
 *
 * <p>Assembles a {@link TransformerStack} of {@link GPTTransformerBlock}s
 * (LayerNorm + multi-head self-attention + configurable FFN).
 *
 * @see LlamaTransformerBuilder
 * @see DeepSeekTransformerBuilder
 */
public final class GPTTransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private Supplier<ActivationFunction> ffnActivationFactory = GELU::new;
    private Random rnd;
    private long seed = 42;

    public GPTTransformerBuilder dModel(int dModel)   { this.dModel = dModel;   return this; }
    public GPTTransformerBuilder nHeads(int nHeads)   { this.nHeads = nHeads;   return this; }
    public GPTTransformerBuilder dFF(int dFF)         { this.dFF = dFF;         return this; }
    public GPTTransformerBuilder nLayers(int nLayers) { this.nLayers = nLayers; return this; }

    /** Activation inside the FFN. Default: GELU. */
    public GPTTransformerBuilder ffnActivation(Supplier<ActivationFunction> factory) {
        if (factory == null) throw new IllegalArgumentException("factory must not be null");
        this.ffnActivationFactory = factory;
        return this;
    }

    public GPTTransformerBuilder seed(long seed) { this.seed = seed; this.rnd = null; return this; }

    public GPTTransformerBuilder random(Random rnd) {
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");
        this.rnd = rnd;
        return this;
    }

    public TransformerStack build() {
        TransformerBuilderSupport.validateCommon(dModel, nHeads, dFF, nLayers);

        Random random = (rnd != null) ? rnd : new Random(seed);
        List<Layer> blocks = new ArrayList<>(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.add(new GPTTransformerBlock(dModel, nHeads, dFF, ffnActivationFactory, random));
        }
        return new TransformerStack(blocks);
    }
}

