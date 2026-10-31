package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.ValidationMessageChecks;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class ValidationMessageTest {
    @ParameterizedTest
    @EnumSource(DType.class)
    void legacyInputDiagnosticsSurviveShapeAndStorageGeneralization(DType dtype) {
        ValidationMessageChecks.assertRejections(new CpuBackend(), dtype);
    }

    @Test
    void generalStatisticsRetainTheirOwnDiagnosticInsteadOfNamingTheLayerNormWrapper() {
        var failure = assertThrows(IllegalArgumentException.class,
                () -> NormalizationChecks.statistics(new Tensor(2, 4), new Tensor(2, 2)));
        assertEquals("Normalization denominator shape mismatch", failure.getMessage());
    }
}
