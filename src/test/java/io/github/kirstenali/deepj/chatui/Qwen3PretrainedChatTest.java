package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.qwen3.pretrained", matches = ".+")
class Qwen3PretrainedChatTest {
    @Test void desktopServiceLoadsTheOfficialModelAndGeneratesAnAnswer() throws Exception {
        assertTrue(MetalBackend.isAvailable());
        Tensor.setBackend(new MetalBackend());
        try {
            var service = new DeepJChatService();
            service.loadModel(Path.of(System.getProperty("deepj.qwen3.pretrained")).resolve("config.json"));
            assertEquals("Qwen3", service.getLoadedModelName());
            assertEquals("Hello.", service.generate("Reply with one word: hello", 8, 0.6f, 1, 4));
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }
}
