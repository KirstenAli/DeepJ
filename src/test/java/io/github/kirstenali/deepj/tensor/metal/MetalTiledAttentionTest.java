package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalTiledAttentionTest extends MetalDifferentialTestSupport {
    @BeforeEach
    void requireTiledDevice() { assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE); }

    @ParameterizedTest
    @ValueSource(floats = {0.0f, 0.125f, -0.5f, 2.0f})
    void outputsAndAllGradientsAgreeWithCpu(float scale) {
        compare(cpu, 32, 2, scale, 1.0f);
        compare(cpu, 64, 1, scale, 8.0f);
    }

    @ParameterizedTest
    @CsvSource({"256,1", "288,3", "512,3", "1024,9", "1024,12"})
    void productionShapesAgreeWithTheUnfusedMetalReference(int length, int heads) {
        compare(metal, length, heads, 0.125f, 1.0f);
    }

    private void compare(TensorBackend reference, int length, int heads, float scale, float amplitude) {
        Tensor query = cpu.multiplyScalar(random(length * heads, 64, 11), amplitude);
        Tensor key = cpu.multiplyScalar(random(query.rows, 64, 23), amplitude);
        Tensor value = random(query.rows, 64, 37), gradient = random(query.rows, 64, 43);
        var expected = reference(reference, query, key, value, gradient, heads, scale);
        var saved = metal.causalAttention(query, key, value, heads, scale);
        var actual = metal.causalAttentionBackward(query, key, value, saved, gradient, heads, scale);
        assertTensorClose(expected.output(), saved.output(), 2e-5f, 2e-4f);
        assertGradients(expected.gradients(), actual);
        assertEquals(query.rows * 2, saved.normalization().data.length);
    }

    private static Reference reference(TensorBackend backend, Tensor q, Tensor k, Tensor v,
                                        Tensor gradient, int heads, float scale) {
        Tensor scores = backend.batchedMatmul(q, k, heads, false, true);
        Tensor probabilities = backend.causalSoftmax(scores, q.rows / heads, scale);
        Tensor dp = backend.batchedMatmul(gradient, v, heads, false, true);
        Tensor ds = backend.softmaxBackward(dp, probabilities, scale);
        var gradients = new AttentionGradients(backend.batchedMatmul(ds, k, heads, false, false),
                backend.batchedMatmul(ds, q, heads, true, false),
                backend.batchedMatmul(probabilities, gradient, heads, true, false));
        return new Reference(backend.batchedMatmul(probabilities, v, heads, false, false), gradients);
    }

    private static void assertGradients(AttentionGradients expected, AttentionGradients actual) {
        assertTensorClose(expected.query(), actual.query(), 3e-5f, 5e-4f);
        assertTensorClose(expected.key(), actual.key(), 3e-5f, 5e-4f);
        assertTensorClose(expected.value(), actual.value(), 3e-5f, 5e-4f);
    }

    @Test
    void fullSizeBlocksPreserveReferenceArithmeticBeforeOptimizerAmplification() {
        Tensor q = random(12 * 1024, 64, 229), k = random(q.rows, 64, 233);
        Tensor v = random(q.rows, 64, 239), gradient = random(q.rows, 64, 241);
        var expected = reference(metal, q, k, v, gradient, 12, 0.125f);
        var saved = metal.causalAttention(q, k, v, 12, 0.125f);
        var actual = metal.causalAttentionBackward(q, k, v, saved, gradient, 12, 0.125f);
        assertExact(expected.output(), saved.output());
        assertExact(expected.gradients().query(), actual.query());
        assertExact(expected.gradients().key(), actual.key());
        assertExact(expected.gradients().value(), actual.value());
    }

    private static void assertExact(Tensor expected, Tensor actual) {
        expected.materialize();
        actual.materialize();
        assertArrayEquals(expected.data, actual.data);
    }

    @Test
    void trackedAttentionSurvivesTemporaryCleanupAndDifferentiatesSharedInputs() {
        Tensor input = random(256, 64, 73).requiresGrad(true);
        Tensor upstream = random(256, 64, 79);
        var expected = reference(cpu, input, input, input, upstream, 1, 0.125f).gradients();
        Tensor output = AttentionOps.causal(input, input, input, 1, 0.125f);
        metal.releaseTemporaryResources();
        output.backward(upstream);
        Tensor sum = cpu.add(cpu.add(expected.query(), expected.key()), expected.value());
        assertTensorClose(sum, input.grad(), 6e-5f, 8e-4f);
    }

    @Test
    void inferenceUsesTheFusedPathWithoutSavingAnAutogradGraph() {
        Tensor input = random(256, 64, 83).requiresGrad(true);
        try (var ignored = GradMode.noGrad()) {
            Tensor output = AttentionOps.causal(input, input, input, 1, 0.125f);
            assertFalse(output.requiresGrad());
            assertEquals(256, output.rows);
            output.materialize();
        }
    }

    @Test
    void savedInputsCannotBeMutatedBeforeBackward() {
        Tensor input = random(256, 64, 89).requiresGrad(true);
        Tensor output = AttentionOps.causal(input, input, input, 1, 0.125f);
        try (var ignored = GradMode.noGrad()) {
            input.addScalarInPlace(1.0f);
        }
        assertThrows(IllegalStateException.class, () -> output.backward(Tensor.ones(256, 64)));
    }

    @Test
    void repeatedRunsHaveIdenticalOutputsAndGradients() {
        Tensor q = random(512, 64, 97), k = random(512, 64, 101), v = random(512, 64, 103);
        Tensor gradient = random(512, 64, 107);
        List<Tensor> expected = snapshot(q, k, v, gradient);
        for (int repeat = 0; repeat < 3; repeat++) {
            List<Tensor> actual = snapshot(q, k, v, gradient);
            for (int index = 0; index < expected.size(); index++) {
                assertArrayEquals(expected.get(index).data, actual.get(index).data);
            }
        }
    }

    private List<Tensor> snapshot(Tensor q, Tensor k, Tensor v, Tensor gradient) {
        var saved = metal.causalAttention(q, k, v, 2, 0.125f);
        var result = metal.causalAttentionBackward(q, k, v, saved, gradient, 2, 0.125f);
        return List.of(new Tensor(saved.output()), new Tensor(result.query()),
                new Tensor(result.key()), new Tensor(result.value()));
    }

    @Test
    void futureTokensCannotInfluenceEarlierQueriesOrReceiveTheirGradient() {
        Tensor q = random(256, 64, 109), k = random(256, 64, 113), v = random(256, 64, 127);
        Tensor expected = new Tensor(metal.causalAttention(q, k, v, 1, 0.125f).output());
        Tensor changed = new Tensor(v), gradient = Tensor.zeros(256, 64);
        for (int index = 0; index < changed.data.length; index++) {
            if (index / 64 >= 128) changed.data[index] += 100;
            else gradient.data[index] = 1;
        }
        var saved = metal.causalAttention(q, k, changed, 1, 0.125f);
        var result = metal.causalAttentionBackward(q, k, changed, saved, gradient, 1, 0.125f);
        saved.output().materialize();
        result.key().materialize();
        result.value().materialize();
        assertCausalBoundary(expected, saved.output(), result);
    }

    private static void assertCausalBoundary(Tensor expected, Tensor output, AttentionGradients gradients) {
        for (int index = 0; index < output.data.length; index++) {
            if (index / 64 < 128) assertEquals(expected.data[index], output.data[index]);
            else {
                assertEquals(0.0f, gradients.key().data[index]);
                assertEquals(0.0f, gradients.value().data[index]);
            }
        }
    }

    private record Reference(Tensor output, AttentionGradients gradients) {}
}
