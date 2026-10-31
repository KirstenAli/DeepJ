package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalAttentionOracleTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @CsvSource({"32,1,0.125,1", "64,3,0,1", "64,2,-0.5,1", "64,2,2,8",
            "64,2,2,32", "64,2,-2,32",
            "256,3,0.125,1", "288,3,0.125,1", "512,3,-0.125,1",
            "1024,9,0.125,1", "1024,12,-0.125,1"})
    void forwardAndEveryInputDerivativeAgreeWithIndependentDoubleArithmetic(
            int length, int heads, float scale, float amplitude) {
        assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE);
        Tensor query = cpu.multiplyScalar(random(length * heads, 64, 307), amplitude);
        Tensor key = cpu.multiplyScalar(random(query.matrixRows(), 64, 311), amplitude);
        Tensor value = random(query.matrixRows(), 64, 313), upstream = random(query.matrixRows(), 64, 317);
        var inputs = new AttentionDoubleReference.Inputs(query, key, value, upstream, heads, scale);
        var expected = AttentionDoubleReference.calculate(inputs);
        var saved = metal.causalAttention(query, key, value, heads, scale);
        var actual = metal.causalAttentionBackward(query, key, value, saved, upstream, heads, scale);
        compare("output", expected.output(), saved.output(), 2e-5, 2e-4);
        compare("query", expected.query(), actual.query(), 3e-5, 5e-4);
        compare("key", expected.key(), actual.key(), 3e-5, 5e-4);
        compare("value", expected.value(), actual.value(), 3e-5, 5e-4);
    }

    static void compare(String label, double[] expected, Tensor actual, double absolute, double relative) {
        actual.materialize();
        assertEquals(expected.length, actual.rawData().length);
        double maximum = 0;
        for (int index = 0; index < expected.length; index++) {
            double error = Math.abs(expected[index] - actual.rawData()[index]);
            assertTrue(Float.isFinite(actual.rawData()[index]), label + " nonfinite at " + index);
            assertTrue(error <= absolute + relative * Math.abs(expected[index]),
                    label + " at " + index + " expected=" + expected[index] + " actual=" + actual.rawData()[index]);
            maximum = Math.max(maximum, error);
        }
        System.out.printf("FP64 variable=%s values=%d maxAbs=%.12g%n", label, expected.length, maximum);
    }
}
