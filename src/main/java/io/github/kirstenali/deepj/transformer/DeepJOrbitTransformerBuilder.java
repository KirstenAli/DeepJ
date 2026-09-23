package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJOrbitTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class DeepJOrbitTransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private int maxSeqLen;
    private Random rnd;
    private long seed = 42;

    public DeepJOrbitTransformerBuilder dModel(int dModel) {
        this.dModel = dModel;
        return this;
    }

    public DeepJOrbitTransformerBuilder nHeads(int nHeads) {
        this.nHeads = nHeads;
        return this;
    }

    public DeepJOrbitTransformerBuilder dFF(int dFF) {
        this.dFF = dFF;
        return this;
    }

    public DeepJOrbitTransformerBuilder nLayers(int nLayers) {
        this.nLayers = nLayers;
        return this;
    }

    public DeepJOrbitTransformerBuilder maxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }

    public DeepJOrbitTransformerBuilder seed(long seed) {
        this.seed = seed;
        this.rnd = null;
        return this;
    }

    public DeepJOrbitTransformerBuilder random(Random rnd) {
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");
        this.rnd = rnd;
        return this;
    }

    public TransformerStack build() {
        TransformerBuilderSupport.validateCommon(dModel, nHeads, dFF, nLayers);
        if (maxSeqLen <= 0)  throw new IllegalArgumentException("maxSeqLen must be > 0");

        Random random = (rnd != null) ? rnd : new Random(seed);
        List<Layer> blocks = new ArrayList<>(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.add(new DeepJOrbitTransformerBlock(dModel, nHeads, dFF, maxSeqLen, random));
        }
        return new TransformerStack(blocks);
    }
}
