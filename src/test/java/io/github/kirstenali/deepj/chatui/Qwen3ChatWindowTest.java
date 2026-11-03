package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.assertEquals;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class Qwen3ChatWindowTest extends ChatDesktopTestSupport {
    @TempDir Path directory;

    @Test void loadedQwenSnapshotGeneratesThroughTheDesktopBridgeAndRendersOnlyTheAnswer() throws Exception {
        Tensor.setBackend(new CpuBackend());
        var service = new DeepJChatService();
        service.loadModel(Qwen3ChatFixture.create(directory.resolve("model")).resolve("config.json"));
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view, "setModel('Qwen3');"
                + "chatBridge.sendWithWindowSettings('hello', 4, 0.8, 1, 5, 1, false)"));
        awaitScript(fixture, "!document.getElementById('prompt').disabled");
        verify(fixture, view -> assertEquals("xxxx",
                script(view, "document.querySelector('.assistant .message-content').textContent")));
    }
}
