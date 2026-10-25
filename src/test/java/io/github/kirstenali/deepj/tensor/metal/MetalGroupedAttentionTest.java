package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalGroupedAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @CsvSource({"32,4,1,0.125", "64,6,2,-0.5", "256,9,3,0.125", "288,4,1,0", "1024,9,3,0.125"})
    void compactKernelAndSharedHeadGradientsMatchDoubleReference(
            int length, int queryHeads, int keyValueHeads, float scale) {
        assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE);
        Tensor query = random(length * queryHeads, 64, 401), key = random(length * keyValueHeads, 64, 409);
        Tensor value = random(key.rows, 64, 419), upstream = random(query.rows, 64, 421);
        var expected = AttentionDoubleReference.calculate(new AttentionDoubleReference.Inputs(
                query, key, value, upstream, queryHeads, keyValueHeads, scale));
        var saved = metal.causalAttention(query, key, value, queryHeads, keyValueHeads, scale);
        var actual = metal.causalAttentionBackward(query, key, value, saved, upstream,
                queryHeads, keyValueHeads, scale);
        assertEquals(key.rows, actual.key().rows);
        assertEquals(value.rows, actual.value().rows);
        MetalAttentionOracleTest.compare("output", expected.output(), saved.output(), 2e-5, 2e-4);
        MetalAttentionOracleTest.compare("query", expected.query(), actual.query(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("key", expected.key(), actual.key(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("value", expected.value(), actual.value(), 3e-5, 5e-4);
    }
}
