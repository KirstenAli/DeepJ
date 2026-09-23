package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJPrismTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class DeepJPrismTransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private int maxSeqLen;
    private int qRank;
    private int kvRank;
    private Random rnd;
    private long seed = 42;

    public DeepJPrismTransformerBuilder dModel(int dModel) {
        this.dModel = dModel;
        return this;
    }

    public DeepJPrismTransformerBuilder nHeads(int nHeads) {
        this.nHeads = nHeads;
        return this;
    }

    public DeepJPrismTransformerBuilder dFF(int dFF) {
        this.dFF = dFF;
        return this;
    }

    public DeepJPrismTransformerBuilder nLayers(int nLayers) {
        this.nLayers = nLayers;
        return this;
    }

    public DeepJPrismTransformerBuilder maxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }

    public DeepJPrismTransformerBuilder qRank(int qRank) {
        this.qRank = qRank;
        return this;
    }

    public DeepJPrismTransformerBuilder kvRank(int kvRank) {
        this.kvRank = kvRank;
        return this;
    }

    public DeepJPrismTransformerBuilder seed(long seed) {
        this.seed = seed;
        this.rnd = null;
        return this;
    }

    public DeepJPrismTransformerBuilder random(Random rnd) {
        if (rnd == null) throw new IllegalArgumentException("rnd must not be null");
        this.rnd = rnd;
        return this;
    }

    public TransformerStack build() {
        TransformerBuilderSupport.validateCommon(dModel, nHeads, dFF, nLayers);
        if (maxSeqLen <= 0) throw new IllegalArgumentException("maxSeqLen must be > 0");
        if (qRank <= 0)     throw new IllegalArgumentException("qRank must be > 0");
        if (kvRank <= 0)    throw new IllegalArgumentException("kvRank must be > 0");

        Random random = (rnd != null) ? rnd : new Random(seed);
        List<Layer> blocks = new ArrayList<>(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.add(new DeepJPrismTransformerBlock(dModel, nHeads, qRank, kvRank, dFF, maxSeqLen, random));
        }
        return new TransformerStack(blocks);
    }
}
