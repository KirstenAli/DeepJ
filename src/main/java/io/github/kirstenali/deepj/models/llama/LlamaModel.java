package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.transformer.LlamaTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.Random;

/**
 * Llama-style decoder-only transformer.
 *
 * <p>Architecture differences from {@link io.github.kirstenali.deepj.models.gpt.GPTModel}:
 * <ul>
 *   <li>No learned positional embedding — RoPE is applied inside each attention block.</li>
 *   <li>RMSNorm instead of LayerNorm for the final pre-head normalisation.</li>
 *   <li>SwiGLU feed-forward instead of GELU-FFN.</li>
 * </ul>
 *
 * <p>Forward/backward/parameters are provided by {@link DecoderOnlyModel}.
 */
public final class LlamaModel extends DecoderOnlyModel {

    private final LlamaConfig cfg;

    public LlamaModel(LlamaConfig cfg, long seed) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed)),
                new LlamaTransformerBuilder()
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .maxSeqLen(cfg.maxSeqLen())
                        .seed(seed)
                        .build(),
                new RMSNorm1D(cfg.dModel()),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 1))
        );
        this.cfg = cfg;
    }

    @Override
    public double gradClipNorm() {
        return cfg.gradClipNorm();
    }
}
