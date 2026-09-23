package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

final class MetalBackendTrainingTest extends MetalBackendTestSupport {
    @Test
    void temporaryReleaseKeepsRetainedBuffer() {
        withGpuBackend(MetalBackendTest::assertTemporaryRelease);
    }

    @Test
    void softmaxBackwardMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertSoftmaxBackward);
    }

    @Test
    void softmaxBackwardRejectsShapeMismatch() {
        Tensor gradOutput = randomTensor(2, 3, 21L);
        Tensor softmaxOut = randomTensor(3, 2, 22L);

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.softmaxBackward(gradOutput, softmaxOut));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    @Test
    void layerNormBackwardMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertLayerNormBackward);
    }

    @Test
    void layerNormBackwardRejectsStdShapeMismatch() {
        Tensor dXHat = randomTensor(4, 8, 41L);
        Tensor xHat = randomTensor(4, 8, 42L);
        Tensor badStd = randomTensor(4, 2, 43L);

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.layerNormBackward(dXHat, xHat, badStd, dXHat.cols));
        assertTrue(ex.getMessage().contains("layerNormBackward: std"));
    }

    @Test
    void crossEntropyGradientMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertCrossEntropyGradient);
    }

    @Test
    void crossEntropyGradientRejectsTargetLengthMismatch() {
        Tensor logits = randomTensor(4, 8, 61L);
        int[] badTargets = new int[]{0, 1, 2};

        IllegalArgumentException ex = assertThrows(
                IllegalArgumentException.class,
                () -> gpu.crossEntropyGradient(logits, badTargets));
        assertTrue(ex.getMessage().contains("targets length"));
    }

    @Test
    void adamWUpdateMatchesCpu_singleStep() {
        withGpuBackend(MetalBackendTest::assertSingleAdamStep);
    }

    @Test
    void adamWUpdateMatchesCpu_multipleSteps() {
        withGpuBackend(MetalBackendTest::assertMultipleAdamSteps);
    }


}
