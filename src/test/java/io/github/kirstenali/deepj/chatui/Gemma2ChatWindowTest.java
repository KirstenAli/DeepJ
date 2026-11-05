package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class Gemma2ChatWindowTest extends ChatDesktopTestSupport {
    @Test void gemmaUsesTheExistingDesktopComposerAndStreamingBridge() throws Exception {
        Tensor.setBackend(new CpuBackend());
        var service = new DeepJChatService();
        Path root = Path.of("src/test/resources/gemma2");
        service.loadModel(root.resolve("config.json"));
        assertFalse(service.supportsRollingCache());
        var fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view, "setModel('Gemma 2');"
                + "setRollingCacheAvailable(false);"
                + "chatBridge.sendWithWindowSettings('Hello',4,1,1,2,1,false)"));
        awaitScript(fixture, "!element('prompt').disabled");
        String expected = Qwen3Json.string(Qwen3Json.read(root.resolve("tokenizer-reference.json")).get("chat_reply"));
        verify(fixture, view -> {
            assertEquals(expected, script(view, "document.querySelector('.assistant .message-content').textContent"));
            assertEquals(true, script(view, "element('imageSettings').hidden"));
            assertEquals("Gemma 2", script(view, "element('composerHint').textContent"));
        });
    }
}
