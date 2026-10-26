package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.HashMap;
import java.util.Map;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdIndexGradientTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void repeatedAxisSelectionAndNonleafScatterMatchFiniteDifferences() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 1, 3, 2);
        Tensor upstream = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 1, 3, 2);
        assertTensorGradients(input, value -> value.multiplyScalar(2).indexSelect(1, 2, 0, 2),
                upstream, 1e-3f, 2e-4f, 1e-3f);
    }

    @Test
    void leafSelectionAccumulatesDirectlyAndDoesNotAllocateFullScatterTemporaries() {
        Map<String, Integer> calls = new HashMap<>();
        Tensor.setBackend(counting(calls));
        try {
            Tensor input = Tensor.from(new float[]{1, 2, 3}, 1, 3).requiresGrad(true);
            input.indexSelect(1, 2, 0, 2).sumTensor().backward();
            Tensor first = input.grad();
            input.indexSelect(1, 1).sumTensor().backward();
            assertSame(first, input.grad());
            assertArrayEquals(new float[]{1, 1, 2}, first.toFloatArray());
            assertEquals(2, calls.get("indexAdd"));
            assertNull(calls.get("indexSelectGradient"));
        } finally {
            Tensor.setBackend(new CpuBackend());
        }
    }

    @Test
    void savedGradientBuffersRemainImmutableDuringFurtherIndexedAccumulation() {
        Tensor input = Tensor.from(new float[]{1, 2, 3}).requiresGrad(true);
        input.indexSelect(0, 2, 0, 2).sumTensor().backward();
        Tensor first = input.grad().requiresGrad(true);
        Tensor saved = first.pow(2).sumTensor();
        input.indexSelect(0, 1).sumTensor().backward();
        assertNotSame(first, input.grad());
        assertArrayEquals(new float[]{1, 0, 2}, first.toFloatArray());
        saved.backward();
        assertArrayEquals(new float[]{2, 0, 4}, first.grad().toFloatArray());
    }

    @Test
    void capturedIndicesAndCompactLeafGradientDtypeAreStable() {
        Tensor input = Tensor.from(new float[]{1, 2, 3}, DType.BFLOAT16).requiresGrad(true);
        int[] indices = {2, 0, 2};
        Tensor output = input.indexSelect(0, indices).sumTensor();
        indices[0] = 99;
        output.backward();
        assertArrayEquals(new float[]{1, 0, 2}, input.grad().toFloatArray());
        assertEquals(DType.FLOAT32, input.grad().dtype());
    }

    @Test
    void aliasedUpstreamGradientTakesTheIndependentScatterPath() {
        Tensor input = Tensor.from(new float[]{1, 2}).requiresGrad(true);
        Tensor upstream = Tensor.from(new float[]{1, 2});
        input.gradientState().bindGradient(upstream);
        input.indexSelect(0, 1, 0).backward(upstream);
        assertArrayEquals(new float[]{3, 3}, input.grad().toFloatArray());
    }

    private static TensorBackend counting(Map<String, Integer> calls) {
        var actual = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> {
                    calls.merge(method.getName(), 1, Integer::sum);
                    return method.invoke(actual, arguments);
                });
    }
}
