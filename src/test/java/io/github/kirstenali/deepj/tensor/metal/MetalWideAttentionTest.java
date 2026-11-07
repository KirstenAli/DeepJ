package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalWideAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @CsvSource({"32,4,2", "64,4,1", "96,3,3", "256,4,2"})
    void forwardAndBackwardMatchIndependentDoublePrecision(int length, int heads, int keys) {
        assumeTrue(MetalNative.supportsWideAttention());
        Tensor query = random(length * heads, 128, 307).requiresGrad(true);
        Tensor key = random(length * keys, 128, 311).requiresGrad(true);
        Tensor value = random(length * keys, 128, 313).requiresGrad(true);
        Tensor upstream = random(length * heads, 128, 317);
        float scale = (float) (1 / Math.sqrt(128));
        assertTrue(metal.supportsCausalAttention(query, key, value, heads, keys, scale));
        var expected = AttentionDoubleReference.calculate(
                new AttentionDoubleReference.Inputs(query, key, value, upstream, heads, keys, scale));
        Tensor output = AttentionOps.causal(query, key, value, heads, keys, scale);
        output.backward(upstream);
        MetalAttentionOracleTest.compare("output", expected.output(), output, 2e-5, 2e-4);
        MetalAttentionOracleTest.compare("query", expected.query(), query.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("key", expected.key(), key.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("value", expected.value(), value.grad(), 3e-5, 5e-4);
    }
}
