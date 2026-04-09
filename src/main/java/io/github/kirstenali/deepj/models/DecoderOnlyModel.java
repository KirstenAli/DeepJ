package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.Persistable;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import io.github.kirstenali.deepj.transformer.embeddings.Embedding;
import io.github.kirstenali.deepj.layers.transformer.norm.NormLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared skeleton for decoder-only transformer models:
 * token embedding → block stack → final norm → LM-head.
 *
 * <p>Subclasses pass their concrete stack (a {@link io.github.kirstenali.deepj.transformer.TransformerStack}
 * or any {@link Layer}) to the protected constructor and only need to provide
 * {@link #gradClipNorm()}.
 *
 * <p>Models that add positional embeddings (e.g. GPT) override
 * {@link #embed}, {@link #backwardEmbeddings}, and {@link #embeddingParameters}.
 */
public abstract class DecoderOnlyModel implements CausalLM, Persistable {

    protected final Embedding        tokEmb;
    protected final TransformerStack stack;
    protected final NormLayer        normF;
    protected final Projection       lmHead;

    protected DecoderOnlyModel(Embedding tokEmb, TransformerStack stack, NormLayer normF, Projection lmHead) {
        this.tokEmb = tokEmb;
        this.stack  = stack;
        this.normF  = normF;
        this.lmHead = lmHead;
    }

    // ── Overridable embedding hooks ────────────────────────────────

    /** Maps input ids to the initial hidden state. Override to add positional embeddings. */
    protected Tensor embed(int[] inputIds) {
        return tokEmb.forward(inputIds);
    }

    /** Back-propagates gradient into embedding layer(s). Override to include positional. */
    protected void backwardEmbeddings(Tensor g) {
        tokEmb.backward(g);
    }

    /** Returns all embedding parameters. Override to include positional embedding. */
    protected List<Parameter> embeddingParameters() {
        return new ArrayList<>(tokEmb.parameters());
    }

    // ── Forward ────────────────────────────────────────────────────

    @Override
    public Tensor forward(int[] inputIds) {
        Tensor x = embed(inputIds);
        x = stack.forward(x);
        x = normF.forward(x);
        return lmHead.forward(x);   // logits [seqLen × vocabSize]
    }

    // ── Backward ───────────────────────────────────────────────────

    @Override
    public void backward(Tensor dLogits) {
        Tensor g = lmHead.backward(dLogits);
        g = normF.backward(g);
        g = stack.backward(g);
        backwardEmbeddings(g);
    }

    // ── Parameters ─────────────────────────────────────────────────

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = embeddingParameters();
        ps.addAll(stack.parameters());
        ps.addAll(normF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }
}
