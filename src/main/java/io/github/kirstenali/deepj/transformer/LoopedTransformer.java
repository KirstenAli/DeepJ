package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJNovaTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/** Repeats shared grouped-query attention and SwiGLU blocks; each pass saves its own activations. */
public final class LoopedTransformer implements Layer {

    private final Layer prelude, core, coda;
    private final BiasFreeLinear contextProjection, stateProjection;
    private final RMSNorm1D coreNorm;
    private final int loops;

    public LoopedTransformer(DeepJLoopConfig config, Random random) {
        prelude = stack(config, config.preludeLayers(), 1.0f, random);
        core = stack(config, config.coreLayers(), config.coreResidualScale(), random);
        coda = stack(config, config.codaLayers(), 1.0f, random);
        contextProjection = new BiasFreeLinear(config.dModel(), config.dModel(), random);
        stateProjection = new BiasFreeLinear(config.dModel(), config.dModel(), random);
        coreNorm = new RMSNorm1D(config.dModel());
        loops = config.loops();
    }

    private static Layer stack(DeepJLoopConfig config, int count, float scale, Random random) {
        if (count == 0) return Identity.INSTANCE;
        List<Layer> blocks = new ArrayList<>(count);
        for (int index = 0; index < count; index++) {
            blocks.add(new DeepJNovaTransformerBlock(config.backbone(), random, scale));
        }
        return new TransformerStack(blocks);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor encoded = prelude.forward(input);
        Tensor context = contextProjection.forward(encoded);
        Tensor state = encoded;
        for (int pass = 0; pass < loops; pass++) {
            state = coreNorm.forward(core.forward(stateProjection.forward(state).add(context)));
        }
        return coda.forward(state);
    }

    @Override
    public boolean supportsIncrementalInference() {
        return Stream.of(prelude, core, coda).allMatch(Layer::supportsIncrementalInference);
    }

    @Override
    public Tensor forward(Tensor input, InferenceContext inference) {
        Tensor encoded = prelude.forward(input, inference);
        Tensor context = contextProjection.forward(encoded);
        Tensor state = encoded;
        for (int pass = 0; pass < loops; pass++) {
            state = coreNorm.forward(core.forward(stateProjection.forward(state).add(context), inference));
        }
        return coda.forward(state, inference);
    }

    @Override
    public List<Parameter> parameters() {
        return Stream.of(prelude, core, coda, contextProjection, stateProjection, coreNorm)
                .flatMap(layer -> layer.parameters().stream()).distinct().toList();
    }

    private enum Identity implements Layer {
        INSTANCE;

        @Override
        public Tensor forward(Tensor input) {
            return input;
        }

        @Override
        public boolean supportsIncrementalInference() { return true; }

        @Override
        public List<Parameter> parameters() {
            return List.of();
        }
    }
}
