package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepSeekTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Builder for DeepSeek-style transformer stacks.
 *
 * <p>Assembles a {@link TransformerStack} of {@link DeepSeekTransformerBlock}s
 * (RMSNorm + Multi-Head Latent Attention + SwiGLU).
 *
 * @see GPTTransformerBuilder
 * @see LlamaTransformerBuilder
 */
public final class DeepSeekTransformerBuilder {

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private int maxSeqLen;
    private int qRank;
    private int kvRank;
    private Random rnd;
    private long seed = 42;

    public DeepSeekTransformerBuilder dModel(int dModel)     { this.dModel = dModel;       return this; }
    public DeepSeekTransformerBuilder nHeads(int nHeads)     { this.nHeads = nHeads;       return this; }
    public DeepSeekTransformerBuilder dFF(int dFF)           { this.dFF = dFF;             return this; }
    public DeepSeekTransformerBuilder nLayers(int nLayers)   { this.nLayers = nLayers;     return this; }

    /** Maximum sequence length for the RoPE table. Required. */
    public DeepSeekTransformerBuilder maxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; return this; }

    /** Q latent dimension for MLA. Required. Typical value: {@code dModel / 2}. */
    public DeepSeekTransformerBuilder qRank(int qRank)  { this.qRank = qRank;   return this; }

    /**
     * KV latent dimension for MLA. Required. Typical value: {@code dModel / 4}.
     * Only {@code cKV} (shape {@code seqLen × kvRank}) is cached at inference time.
     */
    public DeepSeekTransformerBuilder kvRank(int kvRank) { this.kvRank = kvRank; return this; }

    public DeepSeekTransformerBuilder seed(long seed) { this.seed = seed; this.rnd = null; return this; }

    public DeepSeekTransformerBuilder random(Random rnd) {
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
            blocks.add(new DeepSeekTransformerBlock(dModel, nHeads, qRank, kvRank, dFF, maxSeqLen, random));
        }
        return new TransformerStack(blocks);
    }
}

