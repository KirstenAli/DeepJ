package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.GPTTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.transformer.embeddings.PositionalEmbedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class GPTModel extends DecoderOnlyModel {

    private final GPTConfig cfg;
    private final PositionalEmbedding posEmb;

    public GPTModel(GPTConfig cfg, long seed) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed)),
                new GPTTransformerBuilder()
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

    public GPTConfig config() {
        return cfg;
    }

}
