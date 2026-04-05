package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepSeekTransformerBlock;
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
 * DeepSeek-style decoder-only transformer.
 *
 * <p>Architecture differences from {@link io.github.kirstenali.deepj.models.llama.LlamaModel}:
 * <ul>
 *   <li>Attention uses {@link DeepSeekTransformerBlock} with Multi-Head Latent Attention (MLA)
 *       instead of standard RoPE-MHA. MLA compresses Q and K/V through low-rank bottlenecks,
 *       significantly reducing KV-cache memory during inference.</li>
 *   <li>All other components (RMSNorm, RoPE, SwiGLU, token embedding, final norm) are identical
 *       to the Llama architecture.</li>
 * </ul>
 *
 * <p>Forward:
 * <pre>
 *   x = tokEmb(ids)                 // [seqLen × dModel]
 *   for each block: x = block(x)    // RMSNorm + MLA + RMSNorm + SwiGLU, with residuals
 *   x = normF(x)
 *   logits = lmHead(x)              // [seqLen × vocabSize]
 * </pre>
 */
public final class DeepSeekModel implements Trainable, Persistable {

    private final DeepSeekConfig cfg;
    private final Embedding tokEmb;
    private final List<DeepSeekTransformerBlock> blocks;
    private final RMSNorm1D normF;
    private final Linear lmHead;

    public DeepSeekModel(DeepSeekConfig cfg, long seed) {
        this.cfg = cfg;
        Random rnd = new Random(seed);

        this.tokEmb = new Embedding(cfg.vocabSize(), cfg.dModel(), rnd);

        this.blocks = new ArrayList<>(cfg.nLayers());
        for (int i = 0; i < cfg.nLayers(); i++) {
            blocks.add(new DeepSeekTransformerBlock(
                    cfg.dModel(), cfg.nHeads(), cfg.qRank(), cfg.kvRank(),
                    cfg.dFF(), cfg.maxSeqLen(), rnd));
        }

        this.normF  = new RMSNorm1D(cfg.dModel());
        this.lmHead = new Linear(cfg.dModel(), cfg.vocabSize(), rnd);
    }

    // ── Forward ────────────────────────────────────────────────────

    public Tensor forward(int[] inputIds) {
        Tensor x = tokEmb.forward(inputIds);
        for (DeepSeekTransformerBlock block : blocks) {
            x = block.forward(x);
        }
        x = normF.forward(x);
        return lmHead.forward(x);       // logits [seqLen × vocabSize]
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
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(tokEmb.parameters());
        for (DeepSeekTransformerBlock b : blocks) ps.addAll(b.parameters());
        ps.addAll(normF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }

    public double gradClipNorm() {
        return cfg.gradClipNorm();
    }
}

