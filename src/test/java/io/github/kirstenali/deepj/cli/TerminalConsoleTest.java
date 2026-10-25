package io.github.kirstenali.deepj.cli;

import org.jline.terminal.Size;
import org.jline.terminal.Terminal;
import org.jline.terminal.impl.DumbTerminal;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class TerminalConsoleTest {
    @Test
    void lineEditorPreservesQuotesBackslashesAndCapitalization() throws Exception {
        String prompt = "Say 'Hello' in C:\\Java\n";
        var input = new ByteArrayInputStream(prompt.getBytes(StandardCharsets.UTF_8));
        var output = new ByteArrayOutputStream();
        try (var console = new TerminalConsole(new DumbTerminal(input, output))) {
            assertEquals(prompt.stripTrailing(), console.readLine("> "));
            assertTrue(console.interactive());
            assertEquals(80, console.width());
            assertEquals(8, console.colors());
            console.clear();
        }
    }

    @Test
    void terminalCapabilitiesAndInterruptHandlerAreRestored() throws Exception {
        var input = new ByteArrayInputStream(new byte[0]);
        var terminal = new DumbTerminal("test", "xterm-256color", input,
                new ByteArrayOutputStream(), StandardCharsets.UTF_8);
        terminal.setSize(new Size(100, 30));
        AtomicInteger count = new AtomicInteger();
        terminal.handle(Terminal.Signal.INT, ignored -> count.addAndGet(10));
        try (var console = new TerminalConsole(terminal)) {
            assertEquals(100, console.width());
            assertEquals(256, console.colors());
            try (var ignored = console.interrupt(count::incrementAndGet)) {
                terminal.raise(Terminal.Signal.INT);
            }
            terminal.raise(Terminal.Signal.INT);
            assertEquals(11, count.get());
            assertNotNull(console.out());
        }
    }
}
