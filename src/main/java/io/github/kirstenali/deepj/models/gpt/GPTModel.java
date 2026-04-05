package io.github.kirstenali.deepj.models.gpt;

import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.training.Trainable;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.GPTTransformerBuilder;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.transformer.embeddings.PositionalEmbedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal GPT-style decoder-only transformer for educational/training use.
 * CPU-only and dependency-free; intended for small models.
 */
public final class GPTModel implements Trainable, Persistable {

    private final GPTConfig cfg;
    private final Random rnd;

    private final Embedding tokEmb;
    private final PositionalEmbedding posEmb;
    private final TransformerStack stack;
    private final LayerNorm1D lnF;
    private final Linear lmHead;

    public GPTModel(GPTConfig cfg, long seed) {
        this.cfg = cfg;
        this.rnd = new Random(seed);

        this.tokEmb = new Embedding(cfg.vocabSize(), cfg.dModel(), rnd);
        this.posEmb = new PositionalEmbedding(cfg.maxSeqLen(), cfg.dModel(), rnd);

        this.stack = new GPTTransformerBuilder()
                .dModel(cfg.dModel())
                .nHeads(cfg.nHeads())
                .dFF(cfg.dFF())
                .nLayers(cfg.nLayers())
                .seed(seed)
                .random(rnd)
                .build();

        this.lnF = new LayerNorm1D(cfg.dModel());
        this.lmHead = new Linear(cfg.dModel(), cfg.vocabSize(), rnd);

        applyInitScale(cfg.initScale());
    }

    public Tensor forward(int[] inputIds) {
        int seqLen = inputIds.length;
        Tensor tok = tokEmb.forward(inputIds);
        Tensor pos = posEmb.forward(seqLen);

        Tensor x = tok.add(pos);
        x = stack.forward(x);
        x = lnF.forward(x);
        return lmHead.forward(x); // logits [seqLen x vocab]
    }

    public void backward(Tensor dLogits) {
        Tensor g = lmHead.backward(dLogits);
        g = lnF.backward(g);
        g = stack.backward(g);

        tokEmb.backward(g);
        posEmb.backward(g);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(tokEmb.parameters());
        ps.addAll(posEmb.parameters());
        ps.addAll(stack.parameters());
        ps.addAll(lnF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }

    public double gradClipNorm() {
        return cfg.gradClipNorm();
    }

    private void applyInitScale(double factor) {
        if (factor == 1.0) return;
        for (Parameter p : parameters()) {
            p.value = p.value.multiplyScalar(factor);
        }
    }
}