package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class DeepJOriginChatServiceTest {

    @TempDir Path directory;
    private TensorBackend previous;

    @BeforeEach
    void usePortableBackend() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }

    @AfterEach
    void restoreBackend() {
        Tensor.setBackend(previous);
    }

    @Test
    void unloadedServiceReportsItsStateAndRejectsGeneration() {
        var service = new DeepJOriginChatService();
        assertFalse(service.isModelLoaded());
        assertEquals("None", service.getLoadedModelName());
        assertThrows(IllegalStateException.class,
                () -> service.generate("Hello", 2, 0.8f, 20, 42L, text -> {}));
    }

    @Test
    void loadsLegacyWeightsAndStreamsOnlyTheContinuation() throws Exception {
        var config = new DeepJOriginConfig(256, 128, 256, 4, 5, 1024);
        Path checkpoint = directory.resolve("origin.bin");
        new DeepJOrigin(config, 42L).save(checkpoint);
        var service = new DeepJOriginChatService();
        service.loadModel(checkpoint);
        assertTrue(service.isModelLoaded());
        assertEquals("origin.bin", service.getLoadedModelName());
        var partials = new ArrayList<String>();
        String result = service.generate("Hello", 2, 0.8f, 20, 42L, partials::add);
        assertFalse(partials.isEmpty());
        assertEquals(result, partials.get(partials.size() - 1));
        assertInvalidRequests(service);
    }

    private static void assertInvalidRequests(ChatService service) {
        assertThrows(IllegalArgumentException.class,
                () -> service.generate(null, 2, 0.8f, 20, 42L, text -> {}));
        assertThrows(IllegalArgumentException.class,
                () -> service.generate(" ", 2, 0.8f, 20, 42L, text -> {}));
        assertThrows(IllegalArgumentException.class,
                () -> service.generate("Hello", 0, 0.8f, 20, 42L, text -> {}));
        assertThrows(IllegalArgumentException.class,
                () -> service.generate("Hello", 2, Float.NaN, 20, 42L, text -> {}));
        assertThrows(IllegalArgumentException.class,
                () -> service.generate("Hello", 2, 0, 20, 42L, text -> {}));
        assertThrows(IllegalArgumentException.class,
                () -> service.generate("Hello", 2, 0.8f, 0, 42L, text -> {}));
    }
}
