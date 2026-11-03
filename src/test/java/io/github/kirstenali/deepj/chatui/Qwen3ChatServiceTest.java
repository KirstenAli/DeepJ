package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3ChatServiceTest {
    @TempDir Path directory;

    @ParameterizedTest @ValueSource(strings = {".", "config.json", "model.safetensors"})
    void existingDesktopServiceRoutesQwenSelectionsAndStreamsFormattedAnswers(String file) throws Exception {
        Tensor.setBackend(new CpuBackend());
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        var service = new DeepJChatService();
        service.loadModel(snapshot.resolve(file));
        assertTrue(service.isModelLoaded());
        assertEquals("Qwen3", service.getLoadedModelName());
        var updates = new ArrayList<String>();
        assertEquals("xxxx", service.generate("hello", 4, 0.8f, 1, 5, 1.1f,
                ContextWindowMode.ROLLING, updates::add));
        assertEquals(java.util.List.of("x", "xx", "xxx", "xxxx"), updates);
    }

    @Test void failedReplacementKeepsTheLoadedModelUsable() throws Exception {
        Tensor.setBackend(new CpuBackend());
        var service = new DeepJChatService();
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        service.loadModel(snapshot);
        Files.writeString(snapshot.resolve("tokenizer.json"), "invalid");
        assertThrows(java.io.IOException.class, () -> service.loadModel(snapshot));
        assertEquals("xx", service.generate("hello", 2, 0.8f, 1, 5));
    }
}
