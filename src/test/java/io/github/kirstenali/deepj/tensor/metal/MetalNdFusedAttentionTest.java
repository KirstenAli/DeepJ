package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalNdFusedAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void batchesAndGroupedHeadsKeepTheirIndependentGradients(DType dtype) {
        assumeTrue(MetalNative.supportsAttention(dtype));
        Tensor q = input(dtype, 4, 673), k = input(dtype, 2, 677), v = input(dtype, 2, 683);
        Tensor gradient = Tensor.from(random(2048, 64, 691).toFloatArray(), 2, 4, 256, 64);
        var expected = AttentionDoubleReference.calculate(new AttentionDoubleReference.Inputs(
                flat(q), flat(k), flat(v), flat(gradient), 8, 4, 0.125f));
        Tensor output = AttentionOps.causal(q, k, v, 4, 2, 0.125f);
        output.backward(gradient);
        assertArrayEquals(q.shape(), output.shape());
        compare(expected, q, k, v);
        assertFalse(TensorTestStorage.hasHostStorage(output));
    }

    private Tensor input(DType dtype, int heads, long seed) {
        Tensor host = Tensor.from(random(512 * heads, 64, seed).toFloatArray(), dtype, 2, heads, 256, 64);
        return metal.copy(host).requiresGrad(true);
    }

    private static Tensor flat(Tensor input) {
        return Tensor.from(input.toFloatArray(), input.numel() / 64, 64);
    }

    private static void compare(AttentionDoubleReference.Result expected, Tensor q, Tensor k, Tensor v) {
        MetalAttentionOracleTest.compare("nd-query", expected.query(), q.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("nd-key", expected.key(), k.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("nd-value", expected.value(), v.grad(), 3e-5, 5e-4);
    }
}
