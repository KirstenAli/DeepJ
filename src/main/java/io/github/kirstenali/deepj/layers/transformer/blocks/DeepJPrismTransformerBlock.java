package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadLatentAttention;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.Random;

public final class DeepJPrismTransformerBlock extends AbstractTransformerBlock {

    public DeepJPrismTransformerBlock(int dModel, int nHeads, int qRank, int kvRank,
                                    int dFF, int maxSeqLen, Random rnd) {
        this(dModel, nHeads, qRank, kvRank, dFF, maxSeqLen, rnd, ParameterFactory.float32());
    }

    public DeepJPrismTransformerBlock(int dModel, int nHeads, int qRank, int kvRank,
                                    int dFF, int maxSeqLen, Random rnd, ParameterFactory parameters) {
        super(new RMSNorm1D(dModel, parameters.scope("firstNorm")),
                new RMSNorm1D(dModel, parameters.scope("secondNorm")),
                new MultiHeadLatentAttention(dModel, nHeads, qRank, kvRank,
                        rotaryEmbedding(dModel, nHeads, maxSeqLen), rnd, parameters.scope("attention")),
                new SwiGLULayer(dModel, dFF, rnd, true, parameters.scope("feedForward")));
    }
}
