package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.attention.GroupedQueryAttention;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ResidualCompatibilityTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void defaultNovaResidualsMatchThePreviousFormulaExactly(boolean normalize) {
        Tensor.setBackend(new CpuBackend());
        var config = new DeepJNovaConfig(11, 4, 8, 4, 2, 1, 12, normalize, 0.2f, 1.0f);
        var block = new DeepJNovaTransformerBlock(config, new Random(503L));
        var reference = reference(config, new Random(503L));
        Tensor input = Tensor.random(4, 8, new Random(509L));
        Tensor gradient = Tensor.random(4, 8, new Random(521L));
        Tensor actual = block.forward(input), expected = reference.forward(input);
        assertArrayEquals(expected.rawData(), actual.rawData(), 0.0f);
        actual.backward(gradient);
        expected.backward(gradient);
        for (int index = 0; index < block.parameters().size(); index++) {
            assertArrayEquals(reference.parameters().get(index).grad().rawData(),
                    block.parameters().get(index).grad().rawData(), 0.0f);
        }
    }

    private static Reference reference(DeepJNovaConfig config, Random random) {
        return new Reference(new RMSNorm1D(config.dModel()), new RMSNorm1D(config.dModel()),
                new GroupedQueryAttention(config, random),
                new SwiGLULayer(config.dModel(), config.dFF(), random, false));
    }

    private record Reference(Layer firstNorm, Layer secondNorm, Layer attention, Layer feedForward) implements Layer {

        @Override
        public Tensor forward(Tensor input) {
            Tensor attended = input.add(attention.forward(firstNorm.forward(input)));
            return attended.add(feedForward.forward(secondNorm.forward(attended)));
        }

        @Override
        public List<Parameter> parameters() {
            return Stream.of(firstNorm, secondNorm, attention, feedForward)
                    .flatMap(layer -> layer.parameters().stream()).toList();
        }
    }
}
