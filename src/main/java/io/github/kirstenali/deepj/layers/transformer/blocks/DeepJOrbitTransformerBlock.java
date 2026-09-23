package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.attention.RoPEMultiHeadSelfAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;

import java.util.Random;

public final class DeepJOrbitTransformerBlock extends AbstractTransformerBlock {

    public DeepJOrbitTransformerBlock(int dModel, int nHeads, int dFF, int maxSeqLen, Random rnd) {
        super(new RMSNorm1D(dModel), new RMSNorm1D(dModel),
                new RoPEMultiHeadSelfAttention(dModel, nHeads, true,
                        rotaryEmbedding(dModel, nHeads, maxSeqLen), rnd),
                new SwiGLULayer(dModel, dFF, rnd));
    }
}
