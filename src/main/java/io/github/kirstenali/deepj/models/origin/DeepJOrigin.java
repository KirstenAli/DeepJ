package io.github.kirstenali.deepj.models.origin;

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
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed)),
                new DeepJOriginTransformerBuilder()
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .seed(seed + 1)
                        .build(),
                new LayerNorm1D(cfg.dModel()),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 2))
        );
        this.cfg    = cfg;
        this.posEmb = new PositionalEmbedding(cfg.maxSeqLen(), cfg.dModel(), new Random(seed + 3));
        applyInitScale(cfg.initScale());
    }

    @Override
    protected Tensor embed(int[] inputIds) {
        return tokEmb.forward(inputIds).add(posEmb.forward(inputIds.length));
    }

    @Override
    protected void backwardEmbeddings(Tensor g) {
        tokEmb.backward(g);
        posEmb.backward(g);
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

    public DeepJOriginConfig config() {
        return cfg;
    }

}
