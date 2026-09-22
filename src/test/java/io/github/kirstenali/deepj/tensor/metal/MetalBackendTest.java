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

public final class MetalBackendTest extends MetalBackendTestSupport {    @Test
    void metalNativeIsAvailable() {
        assertTrue(MetalBackend.isAvailable());
    }

    @Test
    void repeatedTransfersRemainCorrectAfterRelease() {
        for (int iteration = 0; iteration < 32; iteration++) {
            assertTransferAfterRelease(iteration);
        }
    }

    private static void assertTransferAfterRelease(int iteration) {
        Tensor input = randomTensor(64, 64, 200L + iteration);
        Tensor result = gpu.neg(input);
        result.materialize();
        assertEquals(-input.data[0], result.data[0], 1e-6f);
        gpu.releaseResources();
    }

    @Test
    void smallMatmulMatchesExpectedValues() {
        Tensor a = Tensor.from2D(new float[][]{
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}
        });

        Tensor b = Tensor.from2D(new float[][]{
                {7.0f, 8.0f},
                {9.0f, 10.0f},
                {11.0f, 12.0f}
        });

        Tensor c = gpu.matmul(a, b);
        c.materialize();
        assertEquals(2, c.rows);
        assertEquals(2, c.cols);
        assertEquals(58.0f,  c.data[0], 1e-6f);
        assertEquals(64.0f,  c.data[1], 1e-6f);
        assertEquals(139.0f, c.data[1 * 2 + 0], 1e-6f);
        assertEquals(154.0f, c.data[1 * 2 + 1], 1e-6f);
    }

    @Test
    void matmulMatchesCpu_rectangular() {
        Tensor a = randomTensor(128, 192, 1L);
        Tensor b = randomTensor(192, 64, 2L);

        Tensor expected = cpu.matmul(a, b);
        Tensor actual = gpu.matmul(a, b);
        assertTensorClose(expected, actual, 1e-4f, 1e-4f);
    }

    @Test
    void matmulRejectsShapeMismatch() {
        Tensor a = randomTensor(3, 4, 3L);
        Tensor b = randomTensor(5, 2, 4L);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> gpu.matmul(a, b));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    @Test
    void broadcastAndScalarOpsMatchCpu() {
        withGpuBackend(MetalBackendTest::assertBroadcastAndScalarOps);
    }

    @Test
    void reductionsAndTransposeMatchCpu() {
        MetalBackend gpuBackend = new MetalBackend();
        TensorBackend oldBackend = Tensor.backend();
        Tensor.setBackend(gpuBackend);

        try {
            Tensor a = randomTensor(32, 24, 111L);

            assertTensorClose(cpu.transpose(a), gpuBackend.transpose(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumRows(a), gpuBackend.sumRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumAlongRows(a), gpuBackend.sumAlongRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.sumAlongCols(a), gpuBackend.sumAlongCols(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.meanAlongRows(a), gpuBackend.meanAlongRows(a), 1e-4f, 1e-4f);
            assertTensorClose(cpu.varianceAlongRows(a), gpuBackend.varianceAlongRows(a), 1e-4f, 1e-4f);
            assertEquals(cpu.sum(a), gpuBackend.sum(a), 1e-4f);
        } finally {
            gpuBackend.releaseResources();
            Tensor.setBackend(oldBackend);
        }
    }

    @Test
    void maxClampPowAndScatterAddRowsMatchCpu() {
        withGpuBackend(MetalBackendTest::assertMaxClampPowAndScatters);
    }

    @Test
    void scatterAddRowsDuplicateIndicesAtomicModeMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertAtomicScatter);
    }

    @Test
    void scalarSumAbsAndCrossEntropyLossMatchCpu() {
        withGpuBackend(MetalBackendTest::assertScalarLosses);
    }

    @Test
    void globalL2NormMatchesCpu() {
        withGpuBackend(MetalBackendTest::assertGlobalL2Norm);
    }


}
