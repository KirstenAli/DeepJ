package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Repeated device training must re-zero gradients without allocating full host arrays. */
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
        assertDeviceZeros(parameter);
    }

    @Test
    void deviceZeroGradientAccumulatesExactly() {
        Parameter parameter = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        accumulate(parameter, 0.5f);
        parameter.zeroGrad();
        accumulate(parameter, -1.25f);
        assertArrayEquals(new float[]{-1.25f, -1.25f, -1.25f, -1.25f}, parameter.grad().toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void compactAndFloatParametersKeepLazyFloat32GradientsAndAccumulateTwice(DType dtype) {
        Parameter parameter = new Parameter(Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 1, 2));
        assertDeviceZeros(parameter);
        for (int step = 0; step < 3; step++) {
            parameter.zeroGrad();
            assertNull(parameter.grad().rawData());
            accumulate(parameter, 0.5f);
            accumulate(parameter, -1.25f);
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            assertArrayEquals(new float[]{-0.75f, -0.75f, -0.75f, -0.75f}, parameter.grad().toFloatArray());
        }
    }

    @Test
    void nonNullDeviceZeroGradientStillReceivesWeightDecay() {
        Parameter parameter = new Parameter(Tensor.from2D(new float[][]{{10}}));
        parameter.zeroGrad();
        AdamW optimizer = new AdamW(0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);
        optimizer.step(List.of(parameter));
        assertArrayEquals(new float[]{9.9f}, parameter.value.toFloatArray());
        assertNotNull(parameter.grad());
        assertEquals(1, optimizer.stepCount());
    }

    private static void assertDeviceZeros(Parameter parameter) {
        assertInstanceOf(GpuBuffer.class, parameter.grad().getGpuTag());
        assertNull(parameter.grad().rawData());
        assertEquals(DType.FLOAT32, parameter.grad().dtype());
        assertArrayEquals(parameter.value.shape(), parameter.grad().shape());
        assertArrayEquals(new float[parameter.value.numel()], parameter.grad().toFloatArray());
    }

    private static void accumulate(Parameter parameter, float value) {
        parameter.value.multiplyScalar(value).sumTensor().backward();
    }
}
