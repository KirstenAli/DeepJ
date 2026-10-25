package io.github.kirstenali.deepj.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class CliInputTest extends CliTestSupport {
    @TempDir Path directory;

    @Test
    void filePromptsPreserveWhitespaceAndUnicode() throws IOException {
        Path file = directory.resolve("prompt.txt");
        Files.writeString(file, "What is Java?\n😀\n");
        assertEquals(0, DeepJCli.execute(services("", false), "generate", "fake", "--prompt-file", file.toString()));
        assertEquals("What is Java?\n😀\n", model.prompts.get(0));
    }

    @Test
    void rejectsConflictingMissingBlankAndOversizedInputBeforeLoadingWeights() throws IOException {
        Path file = directory.resolve("prompt.txt");
        Files.writeString(file, "x".repeat(1024 * 1024 + 1));
        assertEquals(2, DeepJCli.execute(services("", false), "generate", "fake", "--prompt", ""));
        assertEquals(2, DeepJCli.execute(services("", true), "generate", "fake"));
        assertEquals(2, DeepJCli.execute(services("", false), "generate", "fake", "--prompt-file", file.toString()));
        assertEquals(2, DeepJCli.execute(services("", false), "generate", "fake", "--prompt", "x",
                "--prompt-file", file.toString()));
        assertEquals(1, DeepJCli.execute(services("", false), "generate", "fake", "--prompt-file", "missing-file"));
        assertEquals(0, loads);
    }

    @Test
    void plainConsoleReadsPipedLinesAndNeverClosesTheCallersStreams() throws IOException {
        CliServices services = services("Hello\n", false);
        try (CliConsole plain = CliConsole.system(services)) {
            assertEquals("Hello", plain.readLine("> "));
            assertFalse(plain.interactive());
            assertEquals(80, plain.width());
            assertEquals(0, plain.colors());
            plain.clear();
            plain.interrupt(() -> fail("No native signal expected")).close();
            assertThrows(org.jline.reader.EndOfFileException.class, () -> plain.readLine("> "));
        }
        services.out().print("still open");
        services.out().flush();
        assertTrue(output.toString().endsWith("still open"));
    }

    @Test
    void systemServicesCanBeCreatedWithoutLoadingAModel() {
        CliServices services = CliServices.system();
        assertSame(System.in, services.input());
        assertNotNull(services.environment());
        assertNotNull(services.models());
        assertNotNull(services.consoles());
        assertNotNull(services.inspector());
    }
}
