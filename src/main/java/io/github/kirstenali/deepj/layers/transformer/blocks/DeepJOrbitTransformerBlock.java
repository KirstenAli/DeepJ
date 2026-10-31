package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.attention.RoPEMultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.Random;

public final class DeepJOrbitTransformerBlock extends AbstractTransformerBlock {

    public DeepJOrbitTransformerBlock(int dModel, int nHeads, int dFF, int maxSeqLen, Random rnd) {
        this(dModel, nHeads, dFF, maxSeqLen, rnd, ParameterFactory.float32());
    }

    public DeepJOrbitTransformerBlock(int dModel, int nHeads, int dFF, int maxSeqLen, Random rnd,
                                      ParameterFactory parameters) {
        super(new RMSNorm1D(dModel, parameters.scope("firstNorm")),
                new RMSNorm1D(dModel, parameters.scope("secondNorm")),
                new RoPEMultiHeadSelfAttention(dModel, nHeads, true,
                        rotaryEmbedding(dModel, nHeads, maxSeqLen), rnd, parameters.scope("attention")),
                new SwiGLULayer(dModel, dFF, rnd, true, parameters.scope("feedForward")));
    }
}
