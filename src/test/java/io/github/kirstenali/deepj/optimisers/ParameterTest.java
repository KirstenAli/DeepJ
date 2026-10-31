package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;

public class ParameterTest {

    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }
    @AfterEach void restore() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void zeroGrad_clearsToZeros() {
        Parameter p = new Parameter(Tensor.from2D(new float[][]{{1,2},{3,4}}));
        Tensor originalGrad = p.grad();

        p.grad().rawData()[0] = 7.0f;
        p.grad().rawData()[1 * 2 + 1] = -3.0f;
        p.zeroGrad();

        Assertions.assertSame(originalGrad, p.grad());
        Assertions.assertEquals(0.0f, p.grad().rawData()[0], 1e-12f);
        Assertions.assertEquals(0.0f, p.grad().rawData()[1 * 2 + 1], 1e-12f);
        Assertions.assertEquals(2, p.grad().matrixRows());
        Assertions.assertEquals(2, p.grad().matrixCols());
    }

    @Test
    void zeroGrad_gpuTaggedGrad_replacesBuffer() {
        Parameter p = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        Tensor originalGrad = p.grad();
        p.grad().setGpuTag(new Object());

        p.zeroGrad();

        Assertions.assertNotSame(originalGrad, p.grad());
        Assertions.assertEquals(0.0f, p.grad().rawData()[0], 1e-12f);
        Assertions.assertEquals(0.0f, p.grad().rawData()[1 * 2 + 1], 1e-12f);
    }

    @Test
    void valueRetainsDeviceBuffer() {
        Parameter parameter = new Parameter(new Tensor(2, 3));
        Assertions.assertTrue(parameter.value.retainsDeviceBuffer());
        Assertions.assertFalse(parameter.grad().retainsDeviceBuffer());
    }

    @Test
    void zeroGrad_gpuTaggedGrad_allocatesThroughBackend() {
        Parameter p = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        p.grad().setGpuTag(new Object());
        Tensor replacement = new Tensor(DType.FLOAT32, 2, 2);
        List<Object> requested = new ArrayList<>();
        Tensor.setBackend(zerosSpy(replacement, requested));

        p.zeroGrad();

        Assertions.assertSame(replacement, p.grad());
        Assertions.assertEquals(DType.FLOAT32, requested.get(0));
        Assertions.assertArrayEquals(new int[]{2, 2}, (int[]) requested.get(1));
    }

    @Test
    void zeroGrad_hostlessUntaggedGrad_isReplacedNotReused() {
        Parameter p = new Parameter(Tensor.from2D(new float[][]{{1, 2}, {3, 4}}));
        Tensor hostless = TensorTestStorage.lazy(2, 2);
        p.setGradient(hostless);

        p.zeroGrad();

        Assertions.assertNotSame(hostless, p.grad());
        Assertions.assertNull(hostless.rawData());
        Assertions.assertArrayEquals(new float[]{0, 0, 0, 0}, p.grad().toFloatArray());
    }

    private static TensorBackend zerosSpy(Tensor zeros, List<Object> requested) {
        CpuBackend cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (!method.getName().equals("zeros") || args.length != 2) return method.invoke(cpu, args);
                    requested.addAll(List.of(args));
                    return zeros;
                });
    }
}
