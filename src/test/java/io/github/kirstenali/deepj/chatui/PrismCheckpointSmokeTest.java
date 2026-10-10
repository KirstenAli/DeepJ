package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.modelPath", matches = ".+")
class PrismCheckpointSmokeTest {

    @Test
    void loadsLocalCheckpointAndStreamsOnCpu() throws Exception {
        DeepJChatService service = new DeepJChatService();
        service.loadModel(Path.of(System.getProperty("deepj.modelPath")));
        List<String> updates = new ArrayList<>();
        String result = service.generate("Once upon a time", 4, 0.8f, 40, 7L, updates::add);

        assertTrue(service.isModelLoaded());
        assertFalse(result.isEmpty());
        assertFalse(updates.isEmpty());
    }
}
