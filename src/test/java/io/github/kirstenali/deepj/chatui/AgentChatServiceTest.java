package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class AgentChatServiceTest {
    @TempDir Path directory;

    @Test
    void detectsQwen3AndFallsBackWhenADifferentCheckpointIsLoaded() throws Exception {
        var service = new AgentChatService(request -> fail("Unexpected command"));
        service.loadModel(qwenSnapshot());
        assertEquals(InteractiveTerminal.available(), service.supportsAgent());
        assertEquals("Qwen3", service.getLoadedModelName());
        assertEquals("xxxx", service.generate("Hello", 4, 0.7f, 1, 1L));
        if (InteractiveTerminal.available()) {
            service.enable(true);
            assertEquals("xxxx", service.generate("Hello", 4, 0.7f, 1, 1L));
        }
        Path checkpoint = directory.resolve("model.dj");
        new DeepJOrigin(new DeepJOriginConfig(256, 16, 8, 2, 1, 16), 2).save(checkpoint);
        service.loadModel(checkpoint);
        assertEquals("model.dj", service.getLoadedModelName());
        assertDoesNotThrow(() -> service.generate("Hello", 2, 0.7f, 1, 1L));
        assertFalse(service.supportsAgent());
    }

    private Path qwenSnapshot() throws Exception {
        var snapshot = Qwen3ChatFixture.create(directory.resolve("qwen"));
        Path config = snapshot.resolve("config.json");
        Files.writeString(config, Files.readString(config).replace("\"max_position_embeddings\": 256",
                "\"max_position_embeddings\": 1024"));
        return config;
    }
}
