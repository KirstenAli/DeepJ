package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.testing.ValidationMessageChecks;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class MetalValidationMessageTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void legacyDiagnosticsRejectBeforeAnyNativeAllocationOrMutation(DType dtype) {
        ValidationMessageChecks.assertRejections(metal, dtype);
    }
}
