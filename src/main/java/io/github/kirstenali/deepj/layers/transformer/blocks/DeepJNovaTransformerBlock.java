package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.attention.GroupedQueryAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;

import java.util.Random;

public final class DeepJNovaTransformerBlock extends AbstractTransformerBlock {

    public DeepJNovaTransformerBlock(DeepJNovaConfig config, Random random) {
        super(new RMSNorm1D(config.dModel()), new RMSNorm1D(config.dModel()),
                new GroupedQueryAttention(config, random),
                new SwiGLULayer(config.dModel(), config.dFF(), random, false));
    }
}
