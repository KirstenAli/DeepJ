package io.github.kirstenali.deepj.models.orbit;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.transformer.DeepJOrbitTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.Random;

public final class DeepJOrbitModel extends DecoderOnlyModel {

    private final DeepJOrbitConfig cfg;

    public DeepJOrbitModel(DeepJOrbitConfig cfg, long seed) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed)),
                new DeepJOrbitTransformerBuilder()
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .maxSeqLen(cfg.maxSeqLen())
                        .seed(seed + 1)
                        .build(),
                new RMSNorm1D(cfg.dModel()),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 2))
        );
        this.cfg = cfg;
        applyInitScale(cfg.initScale());
    }

    @Override
    public float gradClipNorm() {
        return cfg.gradClipNorm();
    }

    public DeepJOrbitConfig config() {
        return cfg;
    }
}
