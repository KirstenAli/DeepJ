package io.github.kirstenali.deepj.models.origin;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.DeepJOriginTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.transformer.embeddings.PositionalEmbedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class DeepJOrigin extends DecoderOnlyModel {

    private final DeepJOriginConfig cfg;
    private final PositionalEmbedding posEmb;

    public DeepJOrigin(DeepJOriginConfig cfg, long seed) {
        this(cfg, seed, ParameterFactory.float32());
    }

    public DeepJOrigin(DeepJOriginConfig cfg, long seed, DType dtype) {
        this(cfg, seed, new ParameterFactory(dtype));
    }

    public DeepJOrigin(DeepJOriginConfig cfg, long seed, ParameterFactory parameters) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed), parameters.scope("tokens")),
                new DeepJOriginTransformerBuilder()
                        .parameters(parameters.scope("stack"))
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .seed(seed + 1)
                        .build(),
                new LayerNorm1D(cfg.dModel(), parameters.scope("finalNorm")),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 2), parameters.scope("output"))
        );
        this.cfg    = cfg;
        this.posEmb = new PositionalEmbedding(cfg.maxSeqLen(), cfg.dModel(), new Random(seed + 3),
                parameters.scope("positions"));
        applyInitScale(cfg.initScale());
    }

    @Override
    protected Tensor embed(int[] inputIds) {
        return embed(inputIds, 0);
    }

    @Override
    protected Tensor embed(int[] inputIds, int offset) {
        return tokEmb.forward(inputIds).add(posEmb.forward(inputIds.length, offset));
    }

    @Override
    protected Tensor embedBatch(int[][] inputIds) {
        Tensor embedded = super.embedBatch(inputIds);
        Tensor positions = posEmb.forward(embedded.size(1)).reshape(1, embedded.size(1), cfg.dModel());
        return embedded.add(positions);
    }

    @Override
    protected List<Parameter> embeddingParameters() {
        List<Parameter> ps = new ArrayList<>(tokEmb.parameters());
        ps.addAll(posEmb.parameters());
        return ps;
    }

    @Override
    public float gradClipNorm() {
        return cfg.gradClipNorm();
    }

    @Override
    protected boolean supportsCachedGeneration() { return true; }

    public DeepJOriginConfig config() {
        return cfg;
    }

}
