package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.NdModelFixtures;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ModelOutputOwnershipTest {
    private final TensorBackend previous = Tensor.backend();

    @AfterEach void restore() { Tensor.setBackend(previous); }

    @Test
    void singleSequenceLogitsDoNotEscapeAnUnnecessaryInternalView() {
        Tensor.setBackend(new CpuBackend());
        try (var ignored = GradMode.noGrad()) {
            for (String family : new String[]{"origin", "orbit", "prism", "nova", "loop"}) {
                Tensor output = NdModelFixtures.model(family, 124).forward(new int[]{1, 2, 3, 4});
                assertEquals(2, output.rank());
                assertTrue(output.memory().canReleaseTemporary(), family + " returned an internal public view");
            }
        }
    }

    @Test
    void aViewExplicitlyCreatedByTheCallerStillProtectsTheReturnedStorage() {
        Tensor.setBackend(new CpuBackend());
        try (var ignored = GradMode.noGrad()) {
            Tensor output = NdModelFixtures.model("nova", 125).forward(new int[]{1, 2, 3, 4});
            Tensor view = Tensor.aliasStorage(output);
            assertSame(output.memory(), view.memory());
            assertFalse(output.memory().canReleaseTemporary());
        }
    }
}
