package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class ComparisonContractsTest {
    @BeforeEach
    void cpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @CsvSource({"Equal,0,1", "Greater,0,0", "Less,1,0", "GreaterOrEqual,0,1", "LessOrEqual,1,1"})
    void broadcastComparisonsPreserveIntegerPrecision(String operator, long first, long second) {
        var integers =
                new OnnxValue[] {
                    OnnxValue.integers(new long[] {9007199254740992L, 9007199254740993L}, 2),
                    OnnxValue.integers(new long[] {9007199254740993L}, 1)
                };
        var floats =
                new OnnxValue[] {
                    OnnxValue.of(Tensor.from(new float[] {1, 2}, 2)),
                    OnnxValue.of(Tensor.from(new float[] {2}, 1))
                };
        assertArrayEquals(
                new long[] {first, second}, OperatorCalls.run(operator, integers).get(0).longs());
        assertArrayEquals(
                new long[] {first, second}, OperatorCalls.run(operator, floats).get(0).longs());
    }
}
