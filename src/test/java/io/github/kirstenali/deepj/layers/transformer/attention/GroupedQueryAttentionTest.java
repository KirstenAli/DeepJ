package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.testing.AutogradTestSupport;
import io.github.kirstenali.deepj.nn.transformer.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertLayerGradients;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GroupedQueryAttentionTest {

    @BeforeEach
    void useCpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 4})
    void inputAndParameterGradientsAreCorrectAcrossGroupingRatios(int kvHeads) {
        var config = new DeepJNovaConfig(11, 4, 8, 4, kvHeads, 1, 12);
        var layer = new GroupedQueryAttention(config, new Random(19L));
        Tensor input = Tensor.random(3, 8, new Random(20L));
        Tensor upstream = Tensor.random(3, 8, new Random(21L));
        assertLayerGradients(layer, input, upstream, 1e-3f, 4e-3f, 7e-2f);
    }

    @Test
    void sharedHeadsAccumulateGradientsRatherThanOverwriteThem() {
        var groups = new GroupedHeads(4, 2);
        Tensor values = Tensor.from2D(new float[][]{{1}, {2}, {3}, {4}}).requiresGrad(true);
        Tensor expanded = groups.expand(values, 2);
        assertArrayEquals(new float[]{1, 2, 1, 2, 3, 4, 3, 4}, expanded.rawData());
        expanded.sumTensor().backward();
        assertArrayEquals(new float[]{2, 2, 2, 2}, values.grad().rawData());
    }

    @Test
    void ungroupedUnnormalizedVariantMatchesExistingRotaryAttention() {
        var config = new DeepJNovaConfig(11, 4, 8, 4, 4, 1, 12, false, 1.0f, 1.0f);
        var actual = new GroupedQueryAttention(config, new Random(32L));
        var expected = new RoPEMultiHeadSelfAttention(8, 4, true,
                new RotaryEmbedding(2, 4), new Random(32L));
        Tensor input = Tensor.random(3, 8, new Random(33L));
        Tensor gradient = Tensor.random(3, 8, new Random(34L));
        assertArrayEquals(expected.forward(input).rawData(), actual.forward(input).rawData(), 1e-6f);
        Tensor expectedGradient = AutogradTestSupport.backward(expected::forward, input, gradient);
        Tensor actualGradient = AutogradTestSupport.backward(actual::forward, input, gradient);
        assertArrayEquals(expectedGradient.rawData(), actualGradient.rawData(), 1e-6f);
        for (int index = 0; index < expected.parameters().size(); index++) {
            assertArrayEquals(expected.parameters().get(index).grad().rawData(),
                    actual.parameters().get(index).grad().rawData(), 1e-6f);
        }
    }
}
