package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.transformer.DeepSeekTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.Random;

/**
 * DeepSeek-style decoder-only transformer.
 *
 * <p>Architecture differences from {@link io.github.kirstenali.deepj.models.llama.LlamaModel}:
 * <ul>
 *   <li>Attention uses Multi-Head Latent Attention (MLA): Q and K/V are compressed through
 *       low-rank bottlenecks ({@code qRank} / {@code kvRank}), significantly reducing
 *       KV-cache memory during inference.</li>
 *   <li>All other components (RMSNorm, RoPE, SwiGLU, token embedding, final norm) are identical
 *       to the Llama architecture.</li>
 * </ul>
 *
 * <p>Forward/backward/parameters are provided by {@link DecoderOnlyModel}.
 */
public final class DeepSeekModel extends DecoderOnlyModel {

    private final DeepSeekConfig cfg;

    public DeepSeekModel(DeepSeekConfig cfg, long seed) {
        super(
                new Embedding(cfg.vocabSize(), cfg.dModel(), new Random(seed)),
                new DeepSeekTransformerBuilder()
                        .dModel(cfg.dModel())
                        .nHeads(cfg.nHeads())
                        .dFF(cfg.dFF())
                        .nLayers(cfg.nLayers())
                        .maxSeqLen(cfg.maxSeqLen())
                        .qRank(cfg.qRank())
                        .kvRank(cfg.kvRank())
                        .seed(seed)
                        .build(),
                new RMSNorm1D(cfg.dModel()),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 1))
        );
        this.cfg = cfg;
    }

    @Override
    public float gradClipNorm() {
        return cfg.gradClipNorm();
    }
}
