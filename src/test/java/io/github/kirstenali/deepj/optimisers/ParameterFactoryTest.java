package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class ParameterFactoryTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void explicitStoragePolicyRetainsFp32GradientsAndCanonicalNames(DType dtype) {
        ParameterFactory factory = new ParameterFactory(dtype).scope("stack").scope("0");
        Parameter weight = factory.random("weight", 3, 4, new Random(18));
        assertEquals("stack.0.weight", weight.name());
        assertEquals(dtype, weight.value.dtype());
        assertEquals(dtype.storageBytes(12), weight.value.storageBytes());
        assertEquals(DType.FLOAT32, weight.grad().dtype());
        assertTrue(weight.value.requiresGrad());
        assertTrue(weight.value.retainsDeviceBuffer());
        assertArrayEquals(new float[]{1, 1}, factory.ones("gamma", 1, 2).value.toFloatArray());
        assertArrayEquals(new float[]{0, 0}, factory.zeros("bias", 1, 2).value.toFloatArray());
    }

    @Test
    void mixedDescriptorsResolveByNameRatherThanCreationOrderAndRemainImmutable() {
        int[] shape = {2, 3};
        TensorDescriptor positions = new TensorDescriptor(DType.BFLOAT16, shape);
        ParameterFactory factory = new ParameterFactory(Map.of("positions.weight", positions,
                "tokens.weight", new TensorDescriptor(DType.FLOAT16, shape)));
        shape[0] = 7;
        Parameter token = factory.scope("tokens").random("weight", 2, 3, new Random(1));
        Parameter position = factory.scope("positions").random("weight", 2, 3, new Random(1));
        assertEquals(DType.FLOAT16, token.value.dtype());
        assertEquals(DType.BFLOAT16, position.value.dtype());
        assertThrows(IllegalArgumentException.class, () -> factory.ones("missing", 1, 2));
        assertThrows(IllegalArgumentException.class, () -> factory.scope("tokens").zeros("weight", 3, 2));
    }

    @Test
    void defaultsKeepSeededValuesWhileInvalidPoliciesFailBeforeAllocation() {
        ParameterFactory factory = ParameterFactory.float32();
        assertArrayEquals(Tensor.random(2, 3, new Random(9)).toFloatArray(),
                factory.random("weight", 2, 3, new Random(9)).value.toFloatArray());
        assertThrows(NullPointerException.class, () -> new ParameterFactory((DType) null));
        assertThrows(NullPointerException.class, () -> new ParameterFactory((Map<String, TensorDescriptor>) null));
        assertThrows(IllegalArgumentException.class, () -> factory.scope(""));
        assertThrows(IllegalArgumentException.class, () -> factory.scope("a..b"));
        assertThrows(IllegalArgumentException.class, () -> factory.scope(null));
        assertThrows(IllegalArgumentException.class, () -> factory.zeros("bias", 0, 2));
        assertThrows(IllegalArgumentException.class, () -> factory.ones("gamma", 1, 0));
        assertThrows(NullPointerException.class, () -> new Parameter(null, new Tensor(1)));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void loadingPolicyKeepsTypedNamedStorageWithoutChangingTheDefaultFactory(DType dtype) {
        var original = new ParameterFactory(dtype);
        var random = new Random(12);
        var loading = original.forLoading().scope("stack");
        var weight = loading.random("weight", 2, 3, random);
        assertEquals("stack.weight", weight.name());
        assertEquals(dtype, weight.value.dtype());
        assertNull(weight.value.getGpuTag());
        assertArrayEquals(new float[6], weight.value.toFloatArray());
        assertEquals(new Random(12).nextLong(), random.nextLong());
        assertArrayEquals(new float[]{1, 1}, loading.ones("gamma", 1, 2).value.toFloatArray());
        assertArrayEquals(new float[]{0, 0}, loading.zeros("bias", 1, 2).value.toFloatArray());
        assertTrue(original.random("weight", 2, 3, new Random(12)).value.sumAbs() > 0);
    }

    @Test
    void loadingPolicyPreservesStrictMixedDescriptorValidation() {
        var factory = new ParameterFactory(Map.of("tokens.weight", new TensorDescriptor(DType.BFLOAT16,
                new int[]{2, 3}))).scope("tokens").forLoading();
        assertEquals(DType.BFLOAT16, factory.random("weight", 2, 3, new Random(12)).value.dtype());
        assertThrows(IllegalArgumentException.class, () -> factory.random("other", 2, 3, new Random(12)));
        assertThrows(IllegalArgumentException.class, () -> factory.random("weight", 3, 2, new Random(12)));
    }
}
