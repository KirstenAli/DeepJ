package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Builder for Llama-style transformer stacks.
 *
 * <p>Assembles a {@link TransformerStack} of {@link LlamaTransformerBlock}s
 * (RMSNorm + RoPE attention + SwiGLU).
 *
 * @see GPTTransformerBuilder
 * @see DeepSeekTransformerBuilder
 */
public final class LlamaTransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private int maxSeqLen;
    private Random rnd;
    private long seed = 42;

    public LlamaTransformerBuilder dModel(int dModel)     { this.dModel = dModel;       return this; }
    public LlamaTransformerBuilder nHeads(int nHeads)     { this.nHeads = nHeads;       return this; }
    public LlamaTransformerBuilder dFF(int dFF)           { this.dFF = dFF;             return this; }
    public LlamaTransformerBuilder nLayers(int nLayers)   { this.nLayers = nLayers;     return this; }

    /** Maximum sequence length for the RoPE table. Required. */
    public LlamaTransformerBuilder maxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; return this; }

    public LlamaTransformerBuilder seed(long seed) { this.seed = seed; this.rnd = null; return this; }

    public LlamaTransformerBuilder random(Random rnd) {
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
            blocks.add(new LlamaTransformerBlock(dModel, nHeads, dFF, maxSeqLen, random));
        }
        return new TransformerStack(blocks);
    }
}

