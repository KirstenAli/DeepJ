package io.github.kirstenali.deepj.tensor.metal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.requireMetal", matches = "true")
class CoverageEnvironmentTest {

    @Test
    void fullCoverageRequiresActualMetalInsteadOfSilentlySkipping() {
        assertTrue(MetalBackend.isAvailable(), "Full coverage requires a working Metal device");
        assertNotNull(System.getenv("LLVM_PROFILE_FILE"));
        assertNotNull(System.getenv("DEEPJ_KERNEL_COVERAGE"));
    }
}
