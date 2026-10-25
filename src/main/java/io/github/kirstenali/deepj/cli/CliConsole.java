package io.github.kirstenali.deepj.cli;

import org.jline.reader.EndOfFileException;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

interface CliConsole extends AutoCloseable {
    PrintWriter out();
    String readLine(String prompt);
    boolean interactive();
    int width();
    int colors();
    void clear();
    StopHandler interrupt(Runnable action);
    @Override void close() throws IOException;

    static CliConsole system(CliServices services) throws IOException {
        return services.interactive() ? new TerminalConsole() : new Plain(services);
    }

    interface StopHandler extends AutoCloseable {
        @Override void close();
    }

    final class Plain implements CliConsole {
        private final BufferedReader input;
        private final PrintWriter output;

        Plain(CliServices services) {
            input = new BufferedReader(new InputStreamReader(services.input(), StandardCharsets.UTF_8));
            output = services.out();
        }

        @Override
        public String readLine(String prompt) {
            output.print(prompt);
            output.flush();
            try {
                String line = input.readLine();
                if (line == null) throw new EndOfFileException();
                return line;
            } catch (IOException error) { throw new IllegalStateException("Cannot read terminal input", error); }
        }

        @Override public PrintWriter out() { return output; }
        @Override public boolean interactive() { return false; }
        @Override public int width() { return 80; }
        @Override public int colors() { return 0; }
        @Override public void clear() { output.println(); }
        @Override public StopHandler interrupt(Runnable action) { return () -> {}; }
        @Override public void close() { output.flush(); }
    }
}
