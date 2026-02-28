package io.github.kirstenali.deepj.gpt;

import io.github.kirstenali.deepj.Tensor;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.training.Trainable;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.layers.transformer.LayerNorm1D;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.transformer.embeddings.PositionalEmbedding;
import io.github.kirstenali.deepj.layers.transformer.TransformerBlock;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Minimal GPT-style decoder-only transformer for educational/training use.
 * CPU-only and dependency-free; intended for small models.
 */
public final class GPTModel implements Trainable {

    private final GPTConfig cfg;
    private final Random rnd;

    private final Embedding tokEmb;
    private final PositionalEmbedding posEmb;
    private final TransformerBlock[] blocks;
    private final LayerNorm1D lnF;
    private final Linear lmHead;

    public GPTModel(GPTConfig cfg, long seed) {
        this.cfg = cfg;
        this.rnd = new Random(seed);

        this.tokEmb = new Embedding(cfg.vocabSize(), cfg.dModel(), rnd);
        this.posEmb = new PositionalEmbedding(cfg.maxSeqLen(), cfg.dModel(), rnd);

        this.blocks = new TransformerBlock[cfg.nLayers()];
        for (int i = 0; i < cfg.nLayers(); i++) {
            blocks[i] = new TransformerBlock(cfg.dModel(), cfg.nHeads(), cfg.dFF(), rnd);
        }

        this.lnF = new LayerNorm1D(cfg.dModel());
        this.lmHead = new Linear(cfg.dModel(), cfg.vocabSize(), rnd);
    }

    public Tensor forward(int[] inputIds) {
        int seqLen = inputIds.length;
        Tensor tok = tokEmb.forward(inputIds);
        Tensor pos = posEmb.forward(seqLen);

        Tensor x = tok.add(pos);

        for (TransformerBlock b : blocks) {
            x = b.forward(x);
        }

        x = lnF.forward(x);
        return lmHead.forward(x); // logits [seqLen x vocab]
    }

    public void backward(Tensor dLogits) {
        Tensor g = lmHead.backward(dLogits);
        g = lnF.backward(g);

        for (int i = blocks.length - 1; i >= 0; i--) {
            g = blocks[i].backward(g);
        }

        tokEmb.backward(g);
        posEmb.backward(g);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(tokEmb.parameters());
        ps.addAll(posEmb.parameters());
        for (TransformerBlock b : blocks) ps.addAll(b.parameters());
        ps.addAll(lnF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }
}
