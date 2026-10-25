package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertFalse;

class MetalGroupedAttentionFallbackTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @CsvSource({"1,4,1", "7,9,3", "33,8,2", "255,4,1", "257,6,2"})
    void unfusedCompactAutogradMatchesDoubleReferenceForShortAndRaggedContexts(
            int length, int queryHeads, int keyValueHeads) {
        Tensor query = random(length * queryHeads, 64, 479).requiresGrad(true);
        Tensor key = random(length * keyValueHeads, 64, 487).requiresGrad(true);
        Tensor value = random(key.rows, 64, 491).requiresGrad(true);
        Tensor upstream = random(query.rows, 64, 499);
        assertFalse(metal.supportsCausalAttention(query, key, value, queryHeads, keyValueHeads, 0.125f));
        var expected = AttentionDoubleReference.calculate(new AttentionDoubleReference.Inputs(
                query, key, value, upstream, queryHeads, keyValueHeads, 0.125f));
        Tensor output = AttentionOps.causal(query, key, value, queryHeads, keyValueHeads, 0.125f);
        output.backward(upstream);
        MetalAttentionOracleTest.compare("output", expected.output(), output, 2e-5, 2e-4);
        MetalAttentionOracleTest.compare("query", expected.query(), query.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("key", expected.key(), key.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("value", expected.value(), value.grad(), 3e-5, 5e-4);
    }
}
