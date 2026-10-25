package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertThrows;

class GroupedAttentionReferenceTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @CsvSource({"1,1,1,0.125", "7,9,3,0.125", "11,8,1,-0.5", "5,4,4,0", "3,4,2,2"})
    void compactForwardAndBackwardAgreeWithIndependentDoubleArithmetic(
            int length, int queryHeads, int keyValueHeads, float scale) {
        Tensor query = variable(length * queryHeads, 13), key = variable(length * keyValueHeads, 17);
        Tensor value = variable(key.rows, 19), upstream = variable(query.rows, 23).requiresGrad(false);
        var expected = AttentionDoubleReference.calculate(new AttentionDoubleReference.Inputs(
                query, key, value, upstream, queryHeads, keyValueHeads, scale));
        Tensor output = AttentionOps.causal(query, key, value, queryHeads, keyValueHeads, scale);
        output.backward(upstream);
        MetalAttentionOracleTest.compare("output", expected.output(), output, 2e-5, 2e-4);
        MetalAttentionOracleTest.compare("query", expected.query(), query.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("key", expected.key(), key.grad(), 3e-5, 5e-4);
        MetalAttentionOracleTest.compare("value", expected.value(), value.grad(), 3e-5, 5e-4);
    }

    @ParameterizedTest
    @CsvSource({"0,1,8,4", "4,0,8,4", "4,3,8,6", "4,2,7,4", "4,2,8,6"})
    void invalidGroupingAndSequenceLengthsAreRejected(int queries, int keys, int queryRows, int keyRows) {
        Tensor query = variable(queryRows, 29), key = variable(keyRows, 31);
        assertThrows(IllegalArgumentException.class,
                () -> AttentionOps.causal(query, key, key, queries, keys, 0.125f));
    }

    private static Tensor variable(int rows, long seed) {
        return Tensor.random(rows, 64, new Random(seed)).requiresGrad(true);
    }
}
