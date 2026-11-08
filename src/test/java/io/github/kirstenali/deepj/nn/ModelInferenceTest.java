package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelInferenceTest {
    private final TensorBackend previous = Tensor.backend();

    @AfterEach void restore() { Tensor.setBackend(previous); }

    @Test void evaluatedBlocksCompleteOnceInInferenceAndPreserveOrdinaryTrainingGradients() {
        var cleanups = new AtomicInteger();
        Tensor.setBackend(counting(cleanups));
        Tensor input = Tensor.from(new float[]{1, 2}).requiresGrad(true);
        ModelInference.evaluate(() -> input.multiplyScalar(3)).sumTensor().backward();
        assertArrayEquals(new float[]{3, 3}, input.grad().toFloatArray());
        assertEquals(0, cleanups.get());
        try (var ignored = GradMode.noGrad()) {
            Tensor result = ModelInference.evaluate(() -> ModelMemory.finish(input.multiplyScalar(4)));
            assertArrayEquals(new float[]{4, 8}, result.toFloatArray());
            assertEquals(1, cleanups.get());
            assertFalse(ModelInference.active());
        }
    }

    @Test void scopedInferenceDefersBlockFinishesAndRestoresTheDefaultBoundary() {
        var cleanups = new AtomicInteger();
        Tensor.setBackend(counting(cleanups));
        Tensor input = Tensor.from(new float[]{1, 2});
        try (var ignored = GradMode.noGrad()) {
            ModelMemory.finish(input);
            assertEquals(1, cleanups.get());
            Tensor output = ModelInference.batch(() -> ModelMemory.finish(input.multiplyScalar(3)));
            assertArrayEquals(new float[]{3, 6}, output.toFloatArray());
            assertEquals(1, cleanups.get());
            assertFalse(ModelInference.active());
            ModelMemory.finish(output);
            assertEquals(2, cleanups.get());
        }
    }

    @Test void nestedScopesAndFailuresRestoreTheCallersExecutionPolicy() {
        Tensor.setBackend(new CpuBackend());
        try (var ignored = GradMode.noGrad()) {
            ModelInference.batch(() -> {
                assertTrue(ModelInference.active());
                assertThrows(IllegalArgumentException.class, () -> ModelInference.batch(() -> {
                    throw new IllegalArgumentException("failed");
                }));
                assertTrue(ModelInference.active());
                return Tensor.from(new float[]{1});
            });
        }
        assertFalse(ModelInference.active());
        assertThrows(IllegalStateException.class, () -> ModelInference.batch(() -> Tensor.from(new float[]{1})));
    }

    private static TensorBackend counting(AtomicInteger cleanups) {
        var cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("releaseTemporaryResources")) cleanups.incrementAndGet();
                    try { return method.invoke(cpu, args); }
                    catch (InvocationTargetException failure) { throw failure.getCause(); }
                });
    }
}
