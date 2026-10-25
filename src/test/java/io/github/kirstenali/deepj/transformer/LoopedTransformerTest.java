package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepJNovaTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertLayerGradients;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class LoopedTransformerTest {

    private final DeepJLoopConfig config = new DeepJLoopConfig(7, 3, 4, 2, 1, 1, 1, 1, 2, 6);

    @BeforeEach
    void cpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void inputAndSharedWeightsMatchNumericalGradients() {
        var stack = new LoopedTransformer(config, new Random(13L));
        assertLayerGradients(stack, Tensor.random(3, 4, new Random(21L)),
                Tensor.random(3, 4, new Random(31L)), 1e-3f, 3e-3f, 6e-2f);
    }

    @Test
    void twoPassesMatchAnExplicitlyUnrolledReference() {
        var actual = new LoopedTransformer(config, new Random(13L));
        var reference = reference(new Random(13L));
        Tensor input = Tensor.random(3, 4, new Random(21L));
        Tensor upstream = Tensor.random(3, 4, new Random(31L));
        Tensor expected = reference.forward(input);
        Tensor observed = actual.forward(input);
        assertArrayEquals(expected.rawData(), observed.rawData(), 0.0f);
        expected.backward(upstream);
        observed.backward(upstream);
        assertParameterGradients(reference.parameters(), actual.parameters());
    }

    private Reference reference(Random random) {
        var initial = new DeepJNovaTransformerBlock(config.backbone(), random);
        var core = new DeepJNovaTransformerBlock(config.backbone(), random, config.coreResidualScale());
        var finalBlock = new DeepJNovaTransformerBlock(config.backbone(), random);
        return new Reference(initial, core, finalBlock, new BiasFreeLinear(4, 4, random),
                new BiasFreeLinear(4, 4, random), new RMSNorm1D(4));
    }

    private static void assertParameterGradients(List<Parameter> a, List<Parameter> b) {
        assertEquals(a.size(), b.size());
        for (int index = 0; index < a.size(); index++) {
            assertArrayEquals(a.get(index).value.rawData(), b.get(index).value.rawData(), 0.0f);
            assertArrayEquals(a.get(index).grad().rawData(), b.get(index).grad().rawData(), 0.0f);
        }
    }

    private record Reference(Layer prelude, Layer core, Layer coda, Layer context, Layer state,
                              Layer norm) implements Layer {

        @Override
        public Tensor forward(Tensor input) {
            Tensor encoded = prelude.forward(input);
            Tensor originalContext = context.forward(encoded);
            Tensor first = norm.forward(core.forward(state.forward(encoded).add(originalContext)));
            Tensor second = norm.forward(core.forward(state.forward(first).add(originalContext)));
            return coda.forward(second);
        }

        @Override
        public List<Parameter> parameters() {
            return Stream.of(prelude, core, coda, context, state, norm)
                    .flatMap(layer -> layer.parameters().stream()).toList();
        }
    }
}
