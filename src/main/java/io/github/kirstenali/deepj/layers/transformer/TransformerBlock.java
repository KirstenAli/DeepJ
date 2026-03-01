package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Pre-LN Transformer block:
 *   x = x + Attn(LN(x))
 *   x = x + MLP(LN(x))
 */
public final class TransformerBlock implements Layer {

    private final LayerNorm1D ln1;
    private final LayerNorm1D ln2;
    private final MultiHeadSelfAttention attn;
    private final FeedForward mlp;

    public TransformerBlock(int dModel, int nHeads, int dFF, Random rnd) {
        this(dModel, nHeads, dFF, new GELU(), rnd);
    }

    public TransformerBlock(int dModel, int nHeads, int dFF, ActivationFunction activation, Random rnd) {
        this.ln1 = new LayerNorm1D(dModel);
        this.ln2 = new LayerNorm1D(dModel);
        this.attn = new MultiHeadSelfAttention(dModel, nHeads, true, rnd);
        this.mlp = new FeedForward(dModel, dFF, activation, rnd);
    }

    public Tensor forward(Tensor x) {
        Tensor ln1Out = ln1.forward(x);
        Tensor attnOut = attn.forward(ln1Out);
        Tensor x2 = x.add(attnOut);

        Tensor ln2Out = ln2.forward(x2);
        Tensor mlpOut = mlp.forward(ln2Out);
        return x2.add(mlpOut);
    }

    public Tensor backward(Tensor gradOut) {
        // residual add2: y = x2 + mlpOut
        Tensor gX2 = gradOut;
        Tensor gMlpOut = gradOut;

        Tensor gLn2Out = mlp.backward(gMlpOut);
        Tensor gX2_from_ln2 = ln2.backward(gLn2Out);
        gX2 = gX2.add(gX2_from_ln2);

        // residual add1: x2 = x + attnOut
        Tensor gX = gX2;
        Tensor gAttnOut = gX2;

        Tensor gLn1Out = attn.backward(gAttnOut);
        Tensor gX_from_ln1 = ln1.backward(gLn1Out);

        return gX.add(gX_from_ln1);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        ps.addAll(ln1.parameters());
        ps.addAll(ln2.parameters());
        ps.addAll(attn.parameters());
        ps.addAll(mlp.parameters());
        return ps;
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

}
