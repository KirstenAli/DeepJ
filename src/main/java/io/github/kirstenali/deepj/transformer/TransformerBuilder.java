package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepSeekTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.GPTTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Convenience builder for assembling transformer stacks.
 *
 * <p>Supports three block architectures via {@link BlockType}:
 * <ul>
 *   <li>{@link BlockType#GPT GPT} (default) — {@link GPTTransformerBlock}:
 *       LayerNorm + multi-head self-attention + GELU FFN</li>
 *   <li>{@link BlockType#LLAMA LLAMA} — {@link LlamaTransformerBlock}:
 *       RMSNorm + RoPE attention + SwiGLU. Requires {@link #maxSeqLen(int)}.</li>
 *   <li>{@link BlockType#DEEPSEEK DEEPSEEK} — {@link DeepSeekTransformerBlock}:
 *       RMSNorm + MLA + SwiGLU. Requires {@link #maxSeqLen(int)},
 *       {@link #qRank(int)}, and {@link #kvRank(int)}.</li>
 * </ul>
 *
 * <p>deepj is transformer-oriented: this builder replaces generic network builders.
 */
public final class TransformerBuilder {

    /**
     * Block architecture to use when building the {@link TransformerStack}.
     */
    public enum BlockType { GPT, LLAMA, DEEPSEEK }

    private int dModel;
    private int nHeads;
    private int dFF;
    private int nLayers;
    private BlockType blockType = BlockType.GPT;
    private Supplier<ActivationFunction> ffnActivationFactory = GELU::new;
    private int maxSeqLen;
    private int qRank;
    private int kvRank;
    private Random rnd;
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

    /**
     * Block architecture. Default: {@link BlockType#GPT}.
     *
     * <ul>
     *   <li>{@link BlockType#LLAMA} additionally requires {@link #maxSeqLen(int)}</li>
     *   <li>{@link BlockType#DEEPSEEK} additionally requires {@link #maxSeqLen(int)},
     *       {@link #qRank(int)}, and {@link #kvRank(int)}</li>
     * </ul>
     */
    public TransformerBuilder blockType(BlockType blockType) {
        if (blockType == null) throw new IllegalArgumentException("blockType must not be null");
        this.blockType = blockType;
        return this;
    }

    /**
     * Maximum sequence length for the RoPE table.
     * Required for {@link BlockType#LLAMA} and {@link BlockType#DEEPSEEK}.
     */
    public TransformerBuilder maxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }

    /**
     * Q latent dimension for Multi-Head Latent Attention.
     * Required for {@link BlockType#DEEPSEEK}. Typical value: {@code dModel / 2}.
     */
    public TransformerBuilder qRank(int qRank) {
        this.qRank = qRank;
        return this;
    }

    /**
     * KV latent dimension for Multi-Head Latent Attention.
     * Required for {@link BlockType#DEEPSEEK}. Typical value: {@code dModel / 4}.
     * Only {@code cKV} (shape {@code seqLen × kvRank}) is cached at inference time.
     */
    public TransformerBuilder kvRank(int kvRank) {
        this.kvRank = kvRank;
        return this;
    }

    /** Activation used inside the FFN (GPT blocks only). Default: GELU. */
    public TransformerBuilder ffnActivation(Supplier<ActivationFunction> activationFactory) {
        if (activationFactory == null) {
            throw new IllegalArgumentException("activationFactory must not be null");
        }
        this.ffnActivationFactory = activationFactory;
        return this;
    }

    public TransformerBuilder seed(long seed) {
        this.seed = seed;
        this.rnd = null;
        return this;
    }

    public TransformerBuilder random(Random rnd) {
        if (rnd == null) {
            throw new IllegalArgumentException("rnd must not be null");
        }
        this.rnd = rnd;
        return this;
    }

    public TransformerStack build() {
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be > 0");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be > 0");
        if (dFF <= 0) throw new IllegalArgumentException("dFF must be > 0");
        if (nLayers <= 0) throw new IllegalArgumentException("nLayers must be > 0");

        if (blockType == BlockType.LLAMA || blockType == BlockType.DEEPSEEK) {
            if (maxSeqLen <= 0)
                throw new IllegalArgumentException("maxSeqLen must be > 0 for LLAMA and DEEPSEEK blocks");
        }
        if (blockType == BlockType.DEEPSEEK) {
            if (qRank <= 0)  throw new IllegalArgumentException("qRank must be > 0 for DEEPSEEK blocks");
            if (kvRank <= 0) throw new IllegalArgumentException("kvRank must be > 0 for DEEPSEEK blocks");
        }

        Random random = (rnd != null) ? rnd : new Random(seed);

        List<Layer> blocks = new ArrayList<>(nLayers);
        for (int i = 0; i < nLayers; i++) {
            blocks.add(switch (blockType) {
                case GPT      -> new GPTTransformerBlock(dModel, nHeads, dFF, ffnActivationFactory, random);
                case LLAMA    -> new LlamaTransformerBlock(dModel, nHeads, dFF, maxSeqLen, random);
                case DEEPSEEK -> new DeepSeekTransformerBlock(dModel, nHeads, qRank, kvRank, dFF, maxSeqLen, random);
            });
        }
        return new TransformerStack(blocks);
    }
}