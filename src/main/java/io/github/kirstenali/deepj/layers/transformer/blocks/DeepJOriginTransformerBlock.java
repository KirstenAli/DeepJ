package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.nn.FNN;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.GELU;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.Random;
import java.util.function.Supplier;

public class DeepJOriginTransformerBlock extends AbstractTransformerBlock {

    public DeepJOriginTransformerBlock(int dModel, int nHeads, int dFF, Random rnd) {
        this(dModel, nHeads, dFF, GELU::new, rnd);
    }

    public DeepJOriginTransformerBlock(int dModel, int nHeads, int dFF,
                               Supplier<ActivationFunction> ffnActivationFactory, Random rnd) {
        this(dModel, nHeads, dFF, ffnActivationFactory, rnd, ParameterFactory.float32());
    }

    public DeepJOriginTransformerBlock(int dModel, int nHeads, int dFF,
                               Supplier<ActivationFunction> ffnActivationFactory, Random rnd,
                               ParameterFactory parameters) {
        super(new LayerNorm1D(dModel, parameters.scope("firstNorm")),
                new LayerNorm1D(dModel, parameters.scope("secondNorm")),
                new MultiHeadSelfAttention(dModel, nHeads, true, rnd, parameters.scope("attention")),
                new FNN(dModel, new int[]{dFF}, dModel, ffnActivationFactory, null, rnd,
                        parameters.scope("feedForward")));
    }
}
