package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalCompactFusedAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactLoadsKeepTinyValuesWithoutNarrowingTheUpstreamGradient(DType dtype) {
        assumeTrue(MetalNative.supportsAttention(dtype));
        float tiny = Math.scalb(1f, dtype == DType.FLOAT16 ? -24 : -120);
        Tensor zeros = new Tensor(dtype, 32, 64), values = Tensor.ones(dtype, 32, 64).multiplyScalar(tiny);
        float[] upstream = new float[32 * 64], expected = new float[upstream.length];
        for (int column = 0; column < 64; column++) {
            upstream[31 * 64 + column] = column % 2 == 0 ? 131072 : 1 + Math.scalb(1f, -20);
        }
        for (int index = 0; index < expected.length; index++) { expected[index] = upstream[31 * 64 + index % 64] / 32; }
        var saved = metal.causalAttention(zeros, zeros, values, 1, 0.125f);
        var gradients = metal.causalAttentionBackward(zeros, zeros, values, saved,
                Tensor.from(upstream, 32, 64), 1, 0.125f);
        assertEquals(tiny, saved.accumulatedOutput().get(0, 0));
        assertArrayEquals(expected, gradients.value().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void typedTilesAndEveryDerivativeAgreeWithIndependentDoubleReference(DType dtype) {
        assumeTrue(MetalNative.supportsAttention(dtype));
        verify(dtype, 32, 1, 1, 0.125f);
        verify(dtype, 256, 6, 2, 0.125f);
        verify(dtype, 288, 4, 1, -0.5f);
    }

    private void verify(DType dtype, int length, int heads, int keys, float scale) {
        Tensor q = Tensor.from(random(length * heads, 64, 613).toFloatArray(), dtype, length * heads, 64);
        Tensor k = Tensor.from(random(length * keys, 64, 617).toFloatArray(), dtype, length * keys, 64);
        Tensor v = Tensor.from(random(length * keys, 64, 619).toFloatArray(), dtype, length * keys, 64);
        Tensor upstream = random(length * heads, 64, 631);
        var reference = new AttentionDoubleReference.Inputs(expand(q), expand(k), expand(v), upstream,
                heads, keys, scale);
        var expected = AttentionDoubleReference.calculate(reference);
        var saved = metal.causalAttention(q, k, v, heads, keys, scale);
        var gradients = metal.causalAttentionBackward(q, k, v, saved, upstream, heads, keys, scale);
        assertEquals(dtype, saved.output().dtype());
        assertEquals(DType.FLOAT32, saved.accumulatedOutput().dtype());
        assertArrayEquals(Tensor.from(saved.accumulatedOutput().toFloatArray(), dtype, q.shape()).toFloatArray(),
                saved.output().toFloatArray());
        compare(expected, saved, gradients);
    }

    private static Tensor expand(Tensor input) {
        return Tensor.from(input.toFloatArray(), input.shape());
    }

    private static void compare(AttentionDoubleReference.Result expected, CausalAttentionResult saved,
                                AttentionGradients gradients) {
        MetalAttentionOracleTest.compare("accumulated", expected.output(), saved.accumulatedOutput(), 2e-5, 2e-4);
        MetalAttentionOracleTest.compare("query", expected.query(), gradients.query(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("key", expected.key(), gradients.key(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("value", expected.value(), gradients.value(), 3e-5, 5e-4);
        assertEquals(DType.FLOAT32, gradients.query().dtype());
        assertEquals(DType.FLOAT32, gradients.key().dtype());
        assertEquals(DType.FLOAT32, gradients.value().dtype());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void autogradRetainsUnroundedStateUntilBackward(DType dtype) {
        assumeTrue(MetalNative.supportsAttention(dtype));
        Tensor source = Tensor.from(random(256, 64, 641).toFloatArray(), dtype, 256, 64);
        Tensor query = metal.copy(source).requiresGrad(true);
        Tensor key = metal.copy(source).requiresGrad(true);
        Tensor value = metal.copy(source).requiresGrad(true);
        Tensor output = AttentionOps.causal(query, key, value, 1, 0.125f);
        assertTrue(metal.supportsCausalAttention(query, key, value, 1, 0.125f));
        output.sumTensor().backward();
        assertEquals(DType.FLOAT32, query.grad().dtype());
        assertTrue(Float.isFinite(query.grad().sum()));
        assertFalse(TensorTestStorage.hasHostStorage(value));
    }
}
