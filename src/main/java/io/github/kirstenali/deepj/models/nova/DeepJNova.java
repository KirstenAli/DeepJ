package io.github.kirstenali.deepj.models.nova;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.TiedEmbeddingProjection;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJNovaTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** A decoder with grouped-query attention, optional QK normalization, and tied embeddings. */
public final class DeepJNova extends DecoderOnlyModel {

    private final DeepJNovaConfig config;
    private final List<Parameter> trainable;

    public DeepJNova(DeepJNovaConfig config, long seed) {
        this(config, seed, new Embedding(config.vocabSize(), config.dModel(), new Random(seed)));
    }

    private DeepJNova(DeepJNovaConfig config, long seed, Embedding embedding) {
        super(embedding, stack(config, new Random(seed + 1)), new RMSNorm1D(config.dModel()),
                new TiedEmbeddingProjection(embedding.weight()));
        this.config = config;
        trainable = super.parameters().stream().distinct().toList();
        applyInitScale(config.initScale());
    }

    private static TransformerStack stack(DeepJNovaConfig config, Random random) {
        List<Layer> blocks = new ArrayList<>(config.nLayers());
        for (int layer = 0; layer < config.nLayers(); layer++) {
            blocks.add(new DeepJNovaTransformerBlock(config, random));
        }
        return new TransformerStack(blocks);
    }

    @Override
    public DeepJNovaConfig config() {
        return config;
    }

    @Override
    public float gradClipNorm() {
        return config.gradClipNorm();
    }

    @Override
    protected boolean supportsCachedGeneration() { return true; }

    @Override
    public List<Parameter> parameters() {
        return trainable;
    }
}
