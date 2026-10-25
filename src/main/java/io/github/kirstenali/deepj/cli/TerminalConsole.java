package io.github.kirstenali.deepj.cli;

import org.jline.keymap.KeyMap;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.Reference;
import org.jline.reader.impl.DefaultParser;
import org.jline.reader.impl.completer.StringsCompleter;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.utils.InfoCmp.Capability;

import java.io.IOException;
import java.io.PrintWriter;

final class TerminalConsole implements CliConsole {
    private final Terminal terminal;
    private final LineReader reader;

    TerminalConsole() throws IOException {
        this(TerminalBuilder.builder().name("DeepJ").system(true).nativeSignals(true).dumb(false).build());
    }

    TerminalConsole(Terminal terminal) {
        this.terminal = terminal;
        var commands = new StringsCompleter(ChatCommands.NAMES);
        reader = LineReaderBuilder.builder().terminal(terminal).appName("DeepJ")
                .parser(new DefaultParser().quoteChars(null).escapeChars(null))
                .option(LineReader.Option.DISABLE_EVENT_EXPANSION, true)
                .completer((lineReader, line, candidates) -> {
                    if (line.line().startsWith("/")) commands.complete(lineReader, line, candidates);
                }).build();
        reader.getWidgets().put("deepj-newline", () -> {
            reader.getBuffer().write('\n');
            return true;
        });
        reader.getKeyMaps().get(LineReader.MAIN).bind(new Reference("deepj-newline"), KeyMap.alt('\r'));
    }

    @Override public PrintWriter out() { return terminal.writer(); }
    @Override public String readLine(String prompt) { return reader.readLine(prompt); }
    @Override public boolean interactive() { return true; }
    @Override public int width() { return terminal.getWidth() > 0 ? terminal.getWidth() : 80; }

    @Override
    public int colors() {
        Integer count = terminal.getNumericCapability(Capability.max_colors);
        return count == null ? 8 : count;
    }

    @Override
    public void clear() {
        terminal.puts(Capability.clear_screen);
        terminal.flush();
    }

    @Override
    public StopHandler interrupt(Runnable action) {
        Terminal.SignalHandler previous = terminal.handle(Terminal.Signal.INT, signal -> action.run());
        return () -> terminal.handle(Terminal.Signal.INT, previous);
    }

    @Override public void close() throws IOException { terminal.close(); }
}
