package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;

import static org.junit.jupiter.api.Assertions.*;

class GenerationCacheLifecycleTest {
    private RecordingBackend backend;
    private DeepJOrigin model;

    @BeforeEach
    void prepare() {
        backend = new RecordingBackend();
        Tensor.setBackend(backend.proxy());
        model = new DeepJOrigin(new DeepJOriginConfig(256, 16, 16, 2, 2, 32), 47);
    }

    @Test
    void streamingCancellationReleasesRequestCachesBeforeAnotherRequest() {
        try {
            assertThrows(CancellationException.class, () -> TextGenerator.stream(model, new ByteTokenizer(),
                    model.config(), "hello", 8, 0.8f, 5, 49, text -> Thread.currentThread().interrupt()));
        } finally {
            Thread.interrupted();
        }
        assertFalse(backend.caches.isEmpty());
        assertTrue(backend.caches.stream().noneMatch(Tensor::retainsDeviceBuffer));
        assertNotNull(generate());
        assertTrue(backend.caches.stream().noneMatch(Tensor::retainsDeviceBuffer));
    }

    @Test
    void listenerFailureAlsoReleasesCachesAndLeavesTheModelUsable() {
        assertThrows(IllegalStateException.class, () -> TextGenerator.stream(model, new ByteTokenizer(),
                model.config(), "hello", 8, 0.8f, 5, 49, text -> {
                    throw new IllegalStateException("Listener failed");
                }));
        assertTrue(backend.caches.stream().noneMatch(Tensor::retainsDeviceBuffer));
        assertNotNull(generate());
    }

    @Test
    void cachedAndUncachedSamplingAndStreamingHaveTheSameSeededResults() {
        String expected = TextGenerator.generate(model::forward, 16, new ByteTokenizer(), "hello",
                15, 0.8f, 5, 49, 1.1f);
        assertEquals(expected, generate());
        List<String> updates = new ArrayList<>();
        assertEquals(expected, TextGenerator.stream(model, new ByteTokenizer(), model.config(), "hello",
                15, 0.8f, 5, 49, 1.1f, updates::add));
        assertEquals(expected, updates.get(updates.size() - 1));
    }

    private String generate() {
        return TextGenerator.generate(model, new ByteTokenizer(), model.config(), "hello", 15, 0.8f, 5, 49, 1.1f);
    }

    private static final class RecordingBackend {
        private final List<Tensor> caches = new ArrayList<>();

        private TensorBackend proxy() {
            var cpu = new CpuBackend();
            return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                    new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                        if (method.getName().equals("writeHeadCache")) caches.add((Tensor) args[0]);
                        try { return method.invoke(cpu, args); }
                        catch (InvocationTargetException error) { throw error.getCause(); }
                    });
        }
    }
}
