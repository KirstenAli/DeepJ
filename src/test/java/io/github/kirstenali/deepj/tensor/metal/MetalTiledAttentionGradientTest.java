package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalTiledAttentionGradientTest extends MetalDifferentialTestSupport {
    @BeforeEach
    void requireTiledDevice() { assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE); }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2})
    void tiledBackwardMatchesIndependentInputPerturbations(int variable) {
        List<Tensor> inputs = List.of(random(64, 64, 139), random(64, 64, 149), random(64, 64, 151));
        Tensor upstream = random(64, 64, 157);
        var saved = metal.causalAttention(inputs.get(0), inputs.get(1), inputs.get(2), 2, 0.125f);
        var gradients = metal.causalAttentionBackward(inputs.get(0), inputs.get(1), inputs.get(2),
                saved, upstream, 2, 0.125f);
        Tensor analytic = List.of(gradients.query(), gradients.key(), gradients.value()).get(variable);
        analytic.materialize();
        for (int index : new int[]{0, 63, 1024, 2048, 4095}) {
            double numerical = difference(inputs, variable, index, upstream);
            assertEquals(numerical, analytic.rawData()[index], 5e-4 + Math.abs(numerical) * 2e-2,
                    "variable " + variable + " flat index " + index);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 9})
    void theFirstCausalTokenHasExactlyZeroQueryAndKeyDerivative(int heads) {
        Tensor query = random(32 * heads, 64, 163), key = random(32 * heads, 64, 167);
        Tensor value = random(32 * heads, 64, 173), gradient = Tensor.zeros(32 * heads, 64);
        for (int head = 0; head < heads; head++) {
            for (int column = 0; column < 64; column++) {
                gradient.rawData()[head * 32 * 64 + column] = 0.7f;
            }
        }
        var saved = metal.causalAttention(query, key, value, heads, 0.125f);
        var result = metal.causalAttentionBackward(query, key, value, saved, gradient, heads, 0.125f);
        result.query().materialize();
        result.key().materialize();
        for (int index = 0; index < query.rawData().length; index++) {
            assertEquals(0.0f, result.query().rawData()[index], "query derivative at " + index);
            assertEquals(0.0f, result.key().rawData()[index], "key derivative at " + index);
        }
    }

    private double difference(List<Tensor> inputs, int variable, int index, Tensor upstream) {
        Tensor tensor = inputs.get(variable);
        int row = index / 64, column = index % 64;
        float original = tensor.get(row, column), epsilon = 0.01f;
        try {
            tensor.set(row, column, original + epsilon);
            double plus = objective(inputs, upstream);
            tensor.set(row, column, original - epsilon);
            return (plus - objective(inputs, upstream)) / (2 * epsilon);
        } finally {
            tensor.set(row, column, original);
        }
    }

    private double objective(List<Tensor> inputs, Tensor upstream) {
        Tensor output = metal.causalAttention(inputs.get(0), inputs.get(1), inputs.get(2), 2, 0.125f).output();
        output.materialize();
        double result = 0;
        for (int index = 0; index < output.rawData().length; index++) {
            result += (double) output.rawData()[index] * upstream.rawData()[index];
        }
        return result;
    }
}
