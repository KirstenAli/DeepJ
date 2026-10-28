package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RecordingMetadataTest {
    private TensorBackend previous;

    @BeforeEach void cpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }
    @AfterEach void restore() { Tensor.setBackend(previous); }

    @Test
    void trackingPreservesShortCircuitAndNoGradNullHandling() {
        Tensor tracked = scalar(2).requiresGrad(true), plain = scalar(3);
        assertFalse(Autograd.tracks());
        assertFalse(Autograd.tracks(plain));
        assertTrue(Autograd.tracks(plain, tracked, null));
        assertThrows(NullPointerException.class, () -> Autograd.tracks(null, tracked));
        try (var ignored = GradMode.noGrad()) {
            assertFalse(Autograd.tracks((Tensor[]) null));
            assertFalse(Autograd.tracks(tracked));
        }
        assertTrue(Autograd.tracks(tracked));
    }

    @Test
    void recordingSnapshotsCallerInputAndSavedArrays() {
        Tensor original = scalar(2).requiresGrad(true), replacement = scalar(9).requiresGrad(true);
        Tensor[] inputs = {original}, saved = {original};
        Tensor output = Autograd.recordSaved(scalar(6), gradient -> new Tensor[]{gradient.multiplyScalar(3)},
                inputs, saved);
        inputs[0] = replacement;
        saved[0] = replacement;
        output.backward();
        assertEquals(3, original.grad().item());
        assertNull(replacement.grad());
    }

    @Test
    void duplicateEdgesRemainOrderedAndAccumulateTwice() {
        Tensor input = scalar(2).requiresGrad(true);
        Tensor output = Autograd.record(scalar(4), gradient -> new Tensor[]{gradient, gradient}, input, input);
        output.backward();
        assertEquals(2, input.grad().item());
    }

    @Test
    void invalidLaterInputCannotRegisterReadersOnEarlierInputs() {
        Tensor input = scalar(2).requiresGrad(true);
        assertThrows(NullPointerException.class,
                () -> Autograd.record(scalar(4), gradient -> new Tensor[]{gradient, gradient}, input, null));
        assertFalse(input.gradientState().storageSavedForBackward());
        assertDoesNotThrow(() -> input.requiresGrad(false));
    }

    private static Tensor scalar(float value) { return Tensor.from2D(new float[][]{{value}}); }
}
