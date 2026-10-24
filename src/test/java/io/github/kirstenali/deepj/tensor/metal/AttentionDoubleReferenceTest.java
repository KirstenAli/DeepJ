package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AttentionDoubleReferenceTest {
    @ParameterizedTest
    @CsvSource({"7,0,1", "7,0.125,1", "33,-0.5,1", "33,2,8"})
    void independentDerivativesAgreeWithFiniteDifferences(int length, float scale, float amplitude) {
        var cpu = new CpuBackend();
        Tensor query = cpu.multiplyScalar(cpu.random(length, 64, new Random(607)), amplitude);
        Tensor key = cpu.multiplyScalar(cpu.random(length, 64, new Random(613)), amplitude);
        Tensor value = cpu.random(length, 64, new Random(617));
        Tensor upstream = cpu.random(length, 64, new Random(619));
        var input = new AttentionDoubleReference.Inputs(query, key, value, upstream, 1, scale);
        var result = AttentionDoubleReference.calculate(input);
        check(input, query, result.query());
        check(input, key, result.key());
        check(input, value, result.value());
    }

    private static void check(AttentionDoubleReference.Inputs input, Tensor variable, double[] derivative) {
        int[] indices = {3, variable.data.length / 2 + 7, variable.data.length - 1};
        for (int index : indices) {
            for (double step : new double[]{0.003, 0.001, 0.0003}) {
                double numerical = difference(input, variable, index, step);
                assertEquals(derivative[index], numerical, 1e-7 + 2e-5 * Math.abs(derivative[index]));
            }
        }
    }

    private static double difference(AttentionDoubleReference.Inputs input, Tensor variable,
                                      int index, double step) {
        float original = variable.data[index];
        float upper = (float) (original + step), lower = (float) (original - step);
        try {
            variable.data[index] = upper;
            double positive = objective(input);
            variable.data[index] = lower;
            return (positive - objective(input)) / ((double) upper - lower);
        } finally {
            variable.data[index] = original;
        }
    }

    private static double objective(AttentionDoubleReference.Inputs input) {
        double[] output = AttentionDoubleReference.calculate(input).output();
        double value = 0;
        for (int index = 0; index < output.length; index++) {
            value += output[index] * input.upstream().data[index];
        }
        return value;
    }
}
