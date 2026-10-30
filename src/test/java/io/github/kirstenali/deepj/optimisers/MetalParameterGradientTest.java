package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** Repeated device training must re-zero gradients on the device instead of allocating host arrays. */
final class MetalParameterGradientTest {
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
    void zeroGradAfterDeviceAccumulationStaysOnDevice() {
        Parameter parameter = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        accumulate(parameter, 0.5f);

        parameter.zeroGrad();

        assertInstanceOf(GpuBuffer.class, parameter.grad().getGpuTag());
        assertArrayEquals(new float[]{0, 0, 0, 0}, parameter.grad().toFloatArray());
    }

    @Test
    void deviceZeroGradientAccumulatesExactly() {
        Parameter parameter = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        accumulate(parameter, 0.5f);
        parameter.zeroGrad();

        accumulate(parameter, -1.25f);

        assertArrayEquals(new float[]{-1.25f, -1.25f, -1.25f, -1.25f}, parameter.grad().toFloatArray());
    }

    private static void accumulate(Parameter parameter, float value) {
        parameter.value.multiplyScalar(value).sumTensor().backward();
    }
}


