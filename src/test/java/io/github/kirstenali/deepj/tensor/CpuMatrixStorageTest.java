package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuMatrixStorageTest {
    @BeforeEach void backend() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void allocationPreservesShapeDtypeZerosAndPhysicalSize(DType dtype) {
        Tensor matrix = Tensor.backend().zeros(dtype, 513, 515);
        assertArrayEquals(new int[]{513, 515}, matrix.shape());
        assertEquals(dtype, matrix.dtype());
        assertEquals(513L * 515 * dtype.bytes(), matrix.storageBytes());
        assertArrayEquals(new float[matrix.numel()], matrix.toFloatArray());
        assertEquals(dtype == DType.FLOAT32 || !CpuBlas.AVAILABLE, matrix.isContiguous());
        assertTrue(Tensor.backend().zeros(dtype, 8192, 32).isContiguous());
        assertTrue(Tensor.backend().zeros(dtype, 2, 513, 515).isContiguous());
        assertTrue(Tensor.zeros(dtype, 513, 515).isContiguous());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void checkpointRestorationAndAdamWMatchDenseParametersBitForBit(DType dtype) {
        Parameter packed = loaded(dtype);
        Parameter dense = new Parameter(Tensor.zeros(dtype, 513, 515));
        Tensor source = Tensor.random(dtype, new Random(37), 515, 513);
        TensorEncoding.restoreWeight(packed.value, source, true);
        TensorEncoding.restoreWeight(dense.value, source, true);
        assertArrayEquals(dense.value.toFloatArray(), packed.value.toFloatArray());
        Tensor gradient = Tensor.ones(DType.FLOAT32, 513, 515);
        packed.setGradient(gradient);
        dense.setGradient(gradient);
        AdamW.defaultAdamW(.01f).step(List.of(packed));
        AdamW.defaultAdamW(.01f).step(List.of(dense));
        assertArrayEquals(dense.value.toFloatArray(), packed.value.toFloatArray());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void packedWeightsRemainLeavesAndReceiveLogicalFloat32Gradients(DType dtype) {
        Parameter packed = loaded(dtype);
        TensorEncoding.restoreWeight(packed.value, Tensor.ones(dtype, 515, 513), true);
        Tensor input = Tensor.ones(DType.FLOAT32, 1, 513).requiresGrad(true);
        input.matmul(packed.value).sumTensor().backward();
        assertEquals(DType.FLOAT32, packed.grad().dtype());
        assertArrayEquals(Tensor.ones(DType.FLOAT32, 513, 515).toFloatArray(), packed.grad().toFloatArray());
        assertArrayEquals(Tensor.ones(DType.FLOAT32, 1, 513).multiplyScalar(515).toFloatArray(),
                input.grad().toFloatArray());
    }

    @Test void loadedFloatWeightsUseOneAllocationAndKeepReusableDenseGradients() {
        Parameter packed = loaded(DType.FLOAT32);
        assertEquals(!CpuBlas.AVAILABLE, packed.value.isContiguous());
        assertEquals(513L * 515 * Float.BYTES, packed.value.storageBytes());
        Tensor gradient = packed.grad();
        gradient.set(0, 0, 3);
        packed.zeroGrad();
        assertSame(gradient, packed.grad());
        assertTrue(gradient.isContiguous());
        assertEquals(0, gradient.get(0, 0));
        assertTrue(Tensor.backend().zerosForLoading(DType.FLOAT32, 8192, 32).isContiguous());
        assertTrue(Tensor.backend().zerosForLoading(DType.FLOAT32, 2, 513, 515).isContiguous());
        assertTrue(ParameterFactory.float32().random("weight", 513, 515, new Random(37)).value.isContiguous());
    }

    private static Parameter loaded(DType dtype) {
        var descriptors = Map.of("weight", new TensorDescriptor(dtype, new int[]{513, 515}));
        return new ParameterFactory(descriptors).forLoading().random("weight", 513, 515, new Random(37));
    }
}
