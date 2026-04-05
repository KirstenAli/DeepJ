package io.github.kirstenali.deepj.models.llama;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Llama-style decoder-only transformer.
 *
 * <p>Architecture differences from {@link io.github.kirstenali.deepj.models.gpt.GPTModel}:
 * <ul>
 *   <li>No learned positional embedding — RoPE is applied inside each attention block.</li>
 *   <li>RMSNorm instead of LayerNorm for the final pre-head normalisation.</li>
 *   <li>{@link LlamaTransformerBlock} (RMSNorm + RoPE-Attn + SwiGLU) instead of the
 *       classic GPT block (LayerNorm + vanilla Attn + GELU-FFN).</li>
 * </ul>
 *
 * <p>Forward:
 * <pre>
 *   x = tokEmb(ids)                  // [seqLen × dModel]
 *   for each block: x = block(x)     // RMSNorm + RoPE-Attn + RMSNorm + SwiGLU, with residuals
 *   x = normF(x)
 *   logits = lmHead(x)               // [seqLen × vocabSize]
 * </pre>
 */
public final class LlamaModel implements Trainable, Persistable {

    private final LlamaConfig cfg;
    private final Embedding tokEmb;
    private final List<LlamaTransformerBlock> blocks;
    private final RMSNorm1D normF;
    private final Linear lmHead;

    public LlamaModel(LlamaConfig cfg, long seed) {
        this.cfg = cfg;
        Random rnd = new Random(seed);

        this.tokEmb = new Embedding(cfg.vocabSize(), cfg.dModel(), rnd);

        this.blocks = new ArrayList<>(cfg.nLayers());
        for (int i = 0; i < cfg.nLayers(); i++) {
            blocks.add(new LlamaTransformerBlock(
                    cfg.dModel(), cfg.nHeads(), cfg.dFF(), cfg.maxSeqLen(), rnd));
        }

        this.normF  = new RMSNorm1D(cfg.dModel());
        this.lmHead = new Linear(cfg.dModel(), cfg.vocabSize(), rnd);
    }

    // ── Forward ────────────────────────────────────────────────────

    public Tensor forward(int[] inputIds) {
        Tensor x = tokEmb.forward(inputIds);
        for (LlamaTransformerBlock block : blocks) {
            x = block.forward(x);
        }
        x = normF.forward(x);
        return lmHead.forward(x);           // logits [seqLen × vocabSize]
    }

    // ── Backward ───────────────────────────────────────────────────

    public void backward(Tensor dLogits) {
        Tensor g = lmHead.backward(dLogits);
        g = normF.backward(g);
        for (int i = blocks.size() - 1; i >= 0; i--) {
            g = blocks.get(i).backward(g);
        }
        tokEmb.backward(g);
    }

    // ── Parameters ─────────────────────────────────────────────────

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>(tokEmb.parameters());
        for (LlamaTransformerBlock b : blocks) ps.addAll(b.parameters());
        ps.addAll(normF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }

    public double gradClipNorm() {
        return cfg.gradClipNorm();
    }
}

