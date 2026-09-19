package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.FNN;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.layers.Layer;

import java.util.Random;
import java.util.function.Supplier;

public class GPTTransformerBlock extends AbstractTransformerBlock {

    private final LayerNorm1D ln1;
    private final LayerNorm1D ln2;
    private final MultiHeadSelfAttention attn;
    private final FNN mlp;

    public GPTTransformerBlock(int dModel, int nHeads, int dFF, Random rnd) {
        this(dModel, nHeads, dFF, GELU::new, rnd);
    }

    public GPTTransformerBlock(int dModel, int nHeads, int dFF,
                               Supplier<ActivationFunction> ffnActivationFactory, Random rnd) {
        this.ln1 = new LayerNorm1D(dModel);
        this.ln2 = new LayerNorm1D(dModel);
        this.attn = new MultiHeadSelfAttention(dModel, nHeads, true, rnd);
        this.mlp = new FNN(dModel, new int[]{ dFF }, dModel, ffnActivationFactory, null, rnd);
    }

    @Override
    protected Layer[] subLayers() {
        return new Layer[]{ ln1, ln2, attn, mlp };
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor ln1Out = ln1.forward(x);
        Tensor attnOut = attn.forward(ln1Out);
        Tensor x2 = x.add(attnOut);

        Tensor ln2Out = ln2.forward(x2);
        Tensor mlpOut = mlp.forward(ln2Out);
        return x2.add(mlpOut);
    }

    @Override
    public Tensor backward(Tensor gradOut) {

        Tensor gX2 = gradOut;
        Tensor gMlpOut = gradOut;

        Tensor gLn2Out = mlp.backward(gMlpOut);
        Tensor gX2_from_ln2 = ln2.backward(gLn2Out);
        gX2 = gX2.add(gX2_from_ln2);

        Tensor gX = gX2;
        Tensor gAttnOut = gX2;

        Tensor gLn1Out = attn.backward(gAttnOut);
        Tensor gX_from_ln1 = ln1.backward(gLn1Out);

        return gX.add(gX_from_ln1);
    }
}
