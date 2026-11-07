package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.HalfSplitRotaryChecks;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class HalfSplitRotaryTest {
    @ParameterizedTest @EnumSource(DType.class)
    void directPairsPreserveValuesAndGradients(DType dtype) {
        Tensor.setBackend(new CpuBackend());
        HalfSplitRotaryChecks.compare(dtype, false);
        HalfSplitRotaryChecks.compare(dtype, true);
    }
}
