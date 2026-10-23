package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SingleBatchMatmulTest {
    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void singleBatchFallbackUsesTheExistingBackendMatmul(boolean ta, boolean tb) {
        CpuBackend cpu = new CpuBackend();
        Tensor a = Tensor.random(ta ? 3 : 2, ta ? 2 : 3, new Random(61));
        Tensor b = Tensor.random(tb ? 4 : 3, tb ? 3 : 4, new Random(62));
        AtomicInteger calls = new AtomicInteger();
        Tensor actual = observing(cpu, calls).batchedMatmul(a, b, 1, ta, tb);
        Tensor expected = cpu.matmul(ta ? cpu.transpose(a) : a, tb ? cpu.transpose(b) : b);
        assertEquals(1, calls.get());
        assertEquals(expected.rows, actual.rows);
        assertEquals(expected.cols, actual.cols);
        assertArrayEquals(expected.data, actual.data);
    }

    private static TensorBackend observing(CpuBackend cpu, AtomicInteger calls) {
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> {
                    if (method.isDefault()) return InvocationHandler.invokeDefault(proxy, method, arguments);
                    if (method.getName().equals("matmul")) calls.incrementAndGet();
                    return method.invoke(cpu, arguments);
                });
    }

    @Test
    void singleBatchCpuFallbackMaterializesDeviceInputsBeforeReadingTheirData() {
        Tensor a = Tensor.ones(1, 1), b = Tensor.ones(1, 1);
        a.setGpuTag(2.0f);
        b.setGpuTag(3.0f);
        TensorBackend previous = Tensor.backend();
        try {
            TensorBackend backend = materializing(new CpuBackend());
            Tensor.setBackend(backend);
            assertEquals(6, backend.batchedMatmul(a, b, 1, false, false).item());
        } finally {
            Tensor.setBackend(previous);
        }
    }

    private static TensorBackend materializing(CpuBackend cpu) {
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> {
                    if (method.getName().equals("materializeTensor")) {
                        Tensor input = (Tensor) arguments[0];
                        input.data[0] = (float) input.getGpuTag();
                        input.setGpuTag(null);
                        return null;
                    }
                    if (method.isDefault()) return InvocationHandler.invokeDefault(proxy, method, arguments);
                    return method.invoke(cpu, arguments);
                });
    }
}
