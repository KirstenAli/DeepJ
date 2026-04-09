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

/**
 * Minimal GPT-style decoder-only transformer for educational/training use.
 *
 * <p>Extends {@link DecoderOnlyModel}; the only GPT-specific additions are:
 * <ul>
 *   <li>A learned {@link PositionalEmbedding} added to the token embedding.</li>
 *   <li>An optional weight {@link GPTConfig#initScale() init-scale} applied after construction.</li>
 *   <li>{@link LayerNorm1D} (instead of RMSNorm) as the final normalisation.</li>
 * </ul>
 */
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
                        .seed(seed)
                        .build(),
                new LayerNorm1D(cfg.dModel()),
                new Linear(cfg.dModel(), cfg.vocabSize(), new Random(seed + 1))
        );
        this.cfg    = cfg;
        this.posEmb = new PositionalEmbedding(cfg.maxSeqLen(), cfg.dModel(), new Random(seed + 2));
        applyInitScale(cfg.initScale());
    }

    // ── Positional-embedding hooks ─────────────────────────────────

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

    // ── Gradient clipping ──────────────────────────────────────────

    @Override
    public double gradClipNorm() {
        return cfg.gradClipNorm();
    }

    // ── Init scale (GPT-2 stabilisation trick) ─────────────────────

    private void applyInitScale(double factor) {
        if (factor == 1.0) return;
        for (Parameter p : parameters()) {
            p.value.multiplyScalarInPlace(factor);
        }
    }
}