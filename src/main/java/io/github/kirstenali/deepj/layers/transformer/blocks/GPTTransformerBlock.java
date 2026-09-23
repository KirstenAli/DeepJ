package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.FNN;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;

import java.util.Random;
import java.util.function.Supplier;

public class GPTTransformerBlock extends AbstractTransformerBlock {

    public GPTTransformerBlock(int dModel, int nHeads, int dFF, Random rnd) {
        this(dModel, nHeads, dFF, GELU::new, rnd);
    }

    public GPTTransformerBlock(int dModel, int nHeads, int dFF,
                               Supplier<ActivationFunction> ffnActivationFactory, Random rnd) {
        super(new LayerNorm1D(dModel), new LayerNorm1D(dModel),
                new MultiHeadSelfAttention(dModel, nHeads, true, rnd),
                new FNN(dModel, new int[]{ dFF }, dModel, ffnActivationFactory, null, rnd));
    }
}
