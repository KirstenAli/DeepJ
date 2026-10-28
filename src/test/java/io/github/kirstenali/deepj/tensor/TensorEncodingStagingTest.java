package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.*;
import java.lang.reflect.*;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorEncodingStagingTest {
    @AfterEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void backendCopiesPreserveRawBitsWithoutKeepingPrivateHostStaging(DType dtype) throws IOException {
        Tensor.setBackend(new CpuBackend());
        Tensor original = Tensor.from(new float[]{-0.0f, Float.intBitsToFloat(0x7fc12345),
                Float.POSITIVE_INFINITY, 1.5f}, dtype, 2, 2);
        byte[] expected = encode(original);
        List<Tensor> staging = new ArrayList<>();
        Tensor.setBackend(observing(staging));
        var result = TensorEncoding.readValuesToBackend(new DataInputStream(new ByteArrayInputStream(expected)),
                dtype, 2, 2);
        assertArrayEquals(expected, encode(result));
        assertEquals(1, staging.size());
        assertNotSame(staging.get(0), result);
        assertFalse(staging.get(0).memory().hasHostStorage());
        assertTrue(result.memory().hasHostStorage());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void truncatedPayloadsNeverReachTheBackend(DType dtype) {
        List<Tensor> staging = new ArrayList<>();
        Tensor.setBackend(observing(staging));
        assertThrows(EOFException.class, () -> TensorEncoding.readValuesToBackend(
                new DataInputStream(new ByteArrayInputStream(new byte[0])), dtype, 2, 2));
        assertTrue(staging.isEmpty());
    }

    private static byte[] encode(Tensor tensor) throws IOException {
        var bytes = new ByteArrayOutputStream();
        TensorEncoding.writeValues(new DataOutputStream(bytes), tensor);
        return bytes.toByteArray();
    }

    private static TensorBackend observing(List<Tensor> staging) {
        var cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class},
                (proxy, method, arguments) -> invoke(cpu, method, arguments, staging));
    }

    private static Object invoke(CpuBackend cpu, Method method, Object[] arguments, List<Tensor> staging)
            throws Throwable {
        if (method.getName().equals("copy")) staging.add((Tensor) arguments[0]);
        try { return method.invoke(cpu, arguments); }
        catch (InvocationTargetException failure) { throw failure.getCause(); }
    }
}
