package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.management.ManagementFactory;

import static org.junit.jupiter.api.Assertions.*;

/** Releasing temporaries must not turn a device gradient into a host array that is re-uploaded every update. */
final class MetalGradientCleanupTest {
    private static final int SIDE = 256;
    private static MetalBackend gpu;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        gpu = new MetalBackend();
        Tensor.setBackend(gpu);
    }

    @AfterAll
    static void tearDown() {
        if (gpu != null) gpu.releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void gradientStaysHostlessAcrossCleanupCycles() {
        Parameter parameter = new Parameter(Tensor.zeros(SIDE, SIDE));
        for (int i = 0; i < 3; i++) {
            parameter.zeroGrad();
            parameter.value.multiplyScalar(0.5f).sumTensor().backward();
            assertNull(parameter.grad().rawData());
            assertInstanceOf(GpuBuffer.class, parameter.grad().getGpuTag());
            gpu.releaseTemporaryResources();
        }
    }

    @Test
    void accumulateAfterCleanupAllocatesNoHostCopy() {
        Parameter parameter = new Parameter(Tensor.zeros(SIDE, SIDE));
        update(parameter);
        parameter.zeroGrad();
        Tensor contribution = parameter.value.addScalar(1.0f);

        long before = allocated();
        Tensor.backend().addInPlace(parameter.grad(), contribution);

        assertTrue(allocated() - before < (long) SIDE * SIDE, "accumulate must not copy the gradient on the host");
        assertEquals(1.0f, parameter.grad().toFloatArray()[0]);
    }

    private static void update(Parameter parameter) {
        parameter.zeroGrad();
        parameter.value.multiplyScalar(0.5f).sumTensor().backward();
        gpu.releaseTemporaryResources();
    }

    private static long allocated() {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        return bean.getThreadAllocatedBytes(Thread.currentThread().threadId());
    }
}

