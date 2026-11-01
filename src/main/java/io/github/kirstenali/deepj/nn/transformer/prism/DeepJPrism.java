package io.github.kirstenali.deepj.nn.transformer.prism;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.transformer.DeepJPrismTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.Random;

public final class DeepJPrism extends DecoderOnlyModel {

    @Override
    protected boolean supportsCachedGeneration() { return true; }

    @Override
    protected boolean supportsRollingGeneration() { return true; }

    private final DeepJPrismConfig cfg;

    public DeepJPrism(DeepJPrismConfig cfg, long seed) {
        this(cfg, seed, ParameterFactory.float32());
    }

    public DeepJPrism(DeepJPrismConfig cfg, long seed, DType dtype) {
        this(cfg, seed, new ParameterFactory(dtype));
    }

    public DeepJPrism(DeepJPrismConfig cfg, long seed, ParameterFactory parameters) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed), parameters.scope("tokens")),
                new DeepJPrismTransformerBuilder()
                        .parameters(parameters.scope("stack"))
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .maxSeqLen(cfg.maxSeqLen())
                        .qRank(cfg.qRank())
                        .kvRank(cfg.kvRank())
                        .seed(seed + 1)
                        .build(),
                new RMSNorm1D(cfg.dModel(), parameters.scope("finalNorm")),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 2), parameters.scope("output"))
        );
        this.cfg = cfg;
        applyInitScale(cfg.initScale());
    }

    @Override
    public float gradClipNorm() {
        return cfg.gradClipNorm();
    }

    public DeepJPrismConfig config() {
        return cfg;
    }
}
