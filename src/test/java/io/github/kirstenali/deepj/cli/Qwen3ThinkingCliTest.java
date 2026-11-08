package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import java.io.ByteArrayInputStream;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class Qwen3ThinkingCliTest {
    @TempDir Path directory;

    @Test void thinkingChatLoadsARealQwenSnapshotAndAllowsAnInteractiveToggle() throws Exception {
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        var output = new StringWriter();
        var errors = new StringWriter();
        var services = services(output, errors, "Hello\n/set thinking off\nAgain\n/exit\n");
        assertEquals(0, DeepJCli.execute(services, "chat", snapshot.toString(), "--backend", "cpu",
                "--max-tokens", "4", "--top-k", "1", "--thinking"), errors.toString());
        assertTrue(output.toString().contains("xxxx"));
        assertTrue(output.toString().contains("thinking"));
        assertEquals("", errors.toString());
    }

    @Test void agentLauncherPassesThinkingThroughTheExistingQwenAdapter() throws Exception {
        assumeTrue(InteractiveTerminal.available());
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"));
        Path config = snapshot.resolve("config.json");
        Files.writeString(config, Files.readString(config).replace(
                "\"max_position_embeddings\": 256", "\"max_position_embeddings\": 4096"));
        var output = new StringWriter();
        var errors = new StringWriter();
        var services = services(output, errors, "Hello\n/set thinking off\nAgain\n/exit\n");
        assertEquals(0, AgentCli.execute(services, snapshot.toString(), "--backend", "cpu",
                "--max-tokens", "4", "--top-k", "1", "--thinking"), errors.toString());
        assertTrue(output.toString().contains("xxxx"));
        assertEquals("", errors.toString());
    }

    private static CliServices services(StringWriter output, StringWriter errors, String input) {
        var defaults = Qwen3CliTest.services(output, errors);
        return new CliServices(new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8)), defaults.out(),
                defaults.err(), false, defaults.environment(), defaults.models(), defaults.consoles(),
                defaults.inspector(), defaults.metalAvailable());
    }
}
