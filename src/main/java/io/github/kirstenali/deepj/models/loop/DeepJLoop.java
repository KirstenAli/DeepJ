package io.github.kirstenali.deepj.models.loop;

import io.github.kirstenali.deepj.layers.TiedEmbeddingProjection;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.transformer.LoopedTransformer;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.List;
import java.util.Random;

/** A causal decoder with a shared recurrent core and full backpropagation through every pass. */
public final class DeepJLoop extends DecoderOnlyModel {

    private final DeepJLoopConfig config;
    private final List<Parameter> trainable;

    public DeepJLoop(DeepJLoopConfig config, long seed) {
        this(config, new Embedding(config.vocabSize(), config.dModel(), new Random(seed)), seed + 1);
    }

    private DeepJLoop(DeepJLoopConfig settings, Embedding embedding, long stackSeed) {
        super(embedding, new TransformerStack(List.of(new LoopedTransformer(settings, new Random(stackSeed)))),
                new RMSNorm1D(settings.dModel()), new TiedEmbeddingProjection(embedding.weight()));
        config = settings;
        trainable = super.parameters();
        applyInitScale(settings.initScale());
    }

    @Override
    public DeepJLoopConfig config() {
        return config;
    }

    @Override
    public float gradClipNorm() {
        return config.gradClipNorm();
    }

    @Override
    public List<Parameter> parameters() {
        return trainable;
    }
}
