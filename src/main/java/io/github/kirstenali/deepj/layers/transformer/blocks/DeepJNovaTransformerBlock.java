package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.attention.GroupedQueryAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;

import java.util.Random;

public final class DeepJNovaTransformerBlock extends AbstractTransformerBlock {

    public DeepJNovaTransformerBlock(DeepJNovaConfig config, Random random) {
        this(config, random, 1.0f);
    }

    public DeepJNovaTransformerBlock(DeepJNovaConfig config, Random random, float residualScale) {
        this(config, random, residualScale, ParameterFactory.float32());
    }

    public DeepJNovaTransformerBlock(DeepJNovaConfig config, Random random, float residualScale,
                                     ParameterFactory parameters) {
        super(new RMSNorm1D(config.dModel(), parameters.scope("firstNorm")),
                new RMSNorm1D(config.dModel(), parameters.scope("secondNorm")),
                new GroupedQueryAttention(config, random, parameters.scope("attention")),
                new SwiGLULayer(config.dModel(), config.dFF(), random, false,
                        parameters.scope("feedForward")), residualScale);
    }
}
