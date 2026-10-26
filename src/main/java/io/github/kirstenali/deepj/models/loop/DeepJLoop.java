package io.github.kirstenali.deepj.models.loop;

import io.github.kirstenali.deepj.layers.TiedEmbeddingProjection;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.transformer.LoopedTransformer;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.List;
import java.util.Random;

/** A looped causal decoder reusing grouped-query attention and SwiGLU blocks across passes. */
public final class DeepJLoop extends DecoderOnlyModel {

    private final DeepJLoopConfig config;
    private final List<Parameter> trainable;

    public DeepJLoop(DeepJLoopConfig config, long seed) {
        this(config, seed, ParameterFactory.float32());
    }

    public DeepJLoop(DeepJLoopConfig config, long seed, DType dtype) {
        this(config, seed, new ParameterFactory(dtype));
    }

    public DeepJLoop(DeepJLoopConfig config, long seed, ParameterFactory parameters) {
        this(config, new Embedding(config.vocabSize(), config.dModel(), new Random(seed),
                parameters.scope("tokens")), seed + 1, parameters);
    }

    private DeepJLoop(DeepJLoopConfig settings, Embedding embedding, long stackSeed, ParameterFactory parameters) {
        super(embedding, new TransformerStack(List.of(new LoopedTransformer(settings, new Random(stackSeed),
                parameters.scope("stack")))), new RMSNorm1D(settings.dModel(), parameters.scope("finalNorm")),
                new TiedEmbeddingProjection(embedding.weight()));
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
    protected boolean supportsCachedGeneration() { return true; }

    @Override
    protected boolean supportsRollingGeneration() { return true; }

    @Override
    public List<Parameter> parameters() {
        return trainable;
    }
}
