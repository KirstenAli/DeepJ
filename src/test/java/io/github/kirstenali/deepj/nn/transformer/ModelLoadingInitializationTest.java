package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.NdModelFixtures;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelLoadingInitializationTest {
    @TempDir Path directory;

    @AfterEach void resetBackend() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void restoringACompleteFileNeverScalesDisposableRandomWeights(String family, DType dtype) throws IOException {
        Tensor.setBackend(new CpuBackend());
        var original = NdModelFixtures.model(family, 28, new ParameterFactory(dtype));
        Path file = directory.resolve("model.dj");
        original.save(file);
        Tensor.setBackend(noInitializationBackend());
        var loaded = ModelLoader.load(file);
        assertEquals(original.config(), loaded.config());
        for (int index = 0; index < original.parameters().size(); index++) {
            assertArrayEquals(original.parameters().get(index).value.toFloatArray(),
                    loaded.parameters().get(index).value.toFloatArray());
        }
    }

    private static TensorBackend noInitializationBackend() {
        var cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, arguments) -> invoke(cpu, method, arguments));
    }

    private static Object invoke(CpuBackend cpu, Method method, Object[] arguments) throws Throwable {
        if (method.getName().equals("multiplyScalarInPlace")) {
            throw new AssertionError("Checkpoint loading initialized weights that the file immediately replaces");
        }
        try { return method.invoke(cpu, arguments); }
        catch (InvocationTargetException failure) { throw failure.getCause(); }
    }
}
