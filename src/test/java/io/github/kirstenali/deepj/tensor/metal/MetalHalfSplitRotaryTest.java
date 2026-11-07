package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.testing.HalfSplitRotaryChecks;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class MetalHalfSplitRotaryTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(DType.class)
    void directPairsPreserveValuesAndGradients(DType dtype) {
        HalfSplitRotaryChecks.compare(dtype, false);
        HalfSplitRotaryChecks.compare(dtype, true);
    }
}
