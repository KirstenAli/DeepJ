package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadLatentAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;

import java.util.Random;

public final class DeepSeekTransformerBlock extends AbstractTransformerBlock {

    public DeepSeekTransformerBlock(int dModel, int nHeads, int qRank, int kvRank,
                                    int dFF, int maxSeqLen, Random rnd) {
        super(new RMSNorm1D(dModel), new RMSNorm1D(dModel),
                new MultiHeadLatentAttention(dModel, nHeads, qRank, kvRank,
                        rotaryEmbedding(dModel, nHeads, maxSeqLen), rnd),
                new SwiGLULayer(dModel, dFF, rnd));
    }
}
