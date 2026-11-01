package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.loop.*;
import io.github.kirstenali.deepj.nn.transformer.nova.*;
import io.github.kirstenali.deepj.nn.transformer.orbit.*;
import io.github.kirstenali.deepj.nn.transformer.origin.*;
import io.github.kirstenali.deepj.nn.transformer.prism.*;

public final class InferenceTestModels {
    public enum Kind { ORIGIN, ORBIT, PRISM, NOVA, LOOP, LOOP_CORE_ONLY }

    private InferenceTestModels() {}

    public static DecoderOnlyModel create(Kind kind, int limit, int width) {
        return switch (kind) {
            case ORIGIN -> new DeepJOrigin(new DeepJOriginConfig(31, limit, width, 2, 2, width * 2), 43);
            case ORBIT -> new DeepJOrbit(new DeepJOrbitConfig(31, limit, width, 2, 2, width * 2), 43);
            case PRISM -> new DeepJPrism(new DeepJPrismConfig(31, limit, width, 2, 2, width * 2, 8, 4), 43);
            case NOVA -> new DeepJNova(new DeepJNovaConfig(31, limit, width, 2, 1, 2, width * 2), 43);
            case LOOP -> loop(limit, width, 1);
            case LOOP_CORE_ONLY -> loop(limit, width, 0);
        };
    }

    private static DeepJLoop loop(int limit, int width, int outer) {
        return new DeepJLoop(new DeepJLoopConfig(31, limit, width, 2, 1, outer, 2, outer, 3, width * 2), 43);
    }
}
