package io.github.kirstenali.deepj.nn.transformer.loop;

import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNovaParameterCount;

public final class DeepJLoopParameterCount {

    private DeepJLoopParameterCount() {}

    public static long count(DeepJLoopConfig config) {
        long width = config.dModel();
        return DeepJNovaParameterCount.count(config.backbone()) + 2L * width * width + width;
    }
}
