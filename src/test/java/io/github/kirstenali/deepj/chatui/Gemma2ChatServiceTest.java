package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Path;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2ChatServiceTest {
    @ParameterizedTest @ValueSource(booleans = {false, true})
    void bothDesktopLaunchersRouteGemmaToPlainChatAndUnloadIt(boolean agentLauncher) throws Exception {
        Tensor.setBackend(new CpuBackend());
        ChatService service = agentLauncher
                ? new AgentChatService(request -> java.util.Optional.empty()) : new DeepJChatService();
        Path root = Path.of("src/test/resources/gemma2");
        service.loadModel(root.resolve("config.json"));
        assertFalse(service.supportsRollingCache());
        if (service instanceof AgentChatService agent) assertFalse(agent.supportsAgent());
        String expected = Qwen3Json.string(Qwen3Json.read(root.resolve("tokenizer-reference.json")).get("chat_reply"));
        assertEquals(expected, service.generate("Hello", 4, 1, 1, 2));
        service.unloadModel();
        assertFalse(service.isModelLoaded());
        assertThrows(IllegalStateException.class, () -> service.generate("Hello", 4, 1, 1, 2));
    }
}
