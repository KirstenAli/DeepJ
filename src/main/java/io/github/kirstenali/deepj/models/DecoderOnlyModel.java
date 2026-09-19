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

    protected Tensor embed(int[] inputIds) {
        return tokEmb.forward(inputIds);
    }

    protected void backwardEmbeddings(Tensor g) {
        tokEmb.backward(g);
    }

    protected List<Parameter> embeddingParameters() {
        return new ArrayList<>(tokEmb.parameters());
    }

    @Override
    public Tensor forward(int[] inputIds) {
        Tensor x = embed(inputIds);
        x = stack.forward(x);
        x = normF.forward(x);
        return lmHead.forward(x);
    }

    @Override
    public void backward(Tensor dLogits) {
        Tensor g = lmHead.backward(dLogits);
        g = normF.backward(g);
        g = stack.backward(g);
        backwardEmbeddings(g);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = embeddingParameters();
        ps.addAll(stack.parameters());
        ps.addAll(normF.parameters());
        ps.addAll(lmHead.parameters());
        return ps;
    }

    protected final void applyInitScale(float factor) {
        if (factor == 1.0f) return;
        for (Parameter parameter : parameters()) {
            if (isRandomWeight(parameter.value)) parameter.value.multiplyScalarInPlace(factor);
        }
    }

    private static boolean isRandomWeight(Tensor tensor) {
        return !isFilledWith(tensor, 0.0f) && !isFilledWith(tensor, 1.0f);
    }

    private static boolean isFilledWith(Tensor tensor, float value) {
        for (float element : tensor.data) if (element != value) return false;
        return true;
    }
}
