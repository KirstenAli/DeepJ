package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.modelPath", matches = ".+")
class PrismCheckpointSmokeTest {

    private TensorBackend previous;

    @BeforeEach
    void selectBackend() {
        previous = Tensor.backend();
        ChatBackend.configure();
    }

    @AfterEach
    void releaseBackend() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @Test
    void loadsLocalCheckpointAndStreamsOnSelectedBackend() throws Exception {
        DeepJChatService service = new DeepJChatService();
        service.loadModel(Path.of(System.getProperty("deepj.modelPath")));
        List<String> updates = new ArrayList<>();
        String result = service.generate("Once upon a time", 4, 0.8f, 40, 7L, updates::add);

        assertTrue(service.isModelLoaded());
        assertFalse(result.isEmpty());
        assertFalse(updates.isEmpty());
    }
}
