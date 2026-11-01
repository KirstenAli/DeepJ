package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import org.jline.reader.EndOfFileException;

import java.io.ByteArrayInputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

abstract class CliTestSupport {
    final StringWriter output = new StringWriter(), errors = new StringWriter();
    final Console console = new Console();
    final Model model = new Model();
    BackendChoice selected;
    int loads;

    CliServices services(String input, boolean interactive) {
        return new CliServices(new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8)),
                new PrintWriter(output, true), new PrintWriter(errors, true), interactive, Map.of(),
                (path, backend) -> {
                    selected = backend;
                    loads++;
                    return model;
                }, ignored -> console, ignored -> info(), () -> false);
    }

    static ModelInspection info() {
        var config = new DeepJPrismConfig(256, 32, 8, 2, 1, 16, 4, 2);
        return new ModelInspection(new ModelSettings(config, PromptFormat.INSTRUCTION_RESPONSE), 1234, 12, 5000);
    }

    final class Console implements CliConsole {
        final ArrayDeque<Object> inputs = new ArrayDeque<>();
        Runnable interrupt = () -> {};
        boolean closed, cleared;

        @Override public PrintWriter out() { return new PrintWriter(output, true); }
        @Override public boolean interactive() { return true; }
        @Override public int width() { return 90; }
        @Override public int colors() { return 256; }
        @Override public void clear() { cleared = true; }
        @Override public void close() { closed = true; }
        @Override public StopHandler interrupt(Runnable action) {
            interrupt = action;
            return () -> interrupt = () -> {};
        }

        @Override public String readLine(String prompt) {
            Object next = inputs.poll();
            if (next == null) throw new EndOfFileException();
            if (next instanceof RuntimeException failure) throw failure;
            return next.toString();
        }
    }

    final class Model implements CliModel {
        final List<String> prompts = new ArrayList<>();
        final List<SamplingSettings> settings = new ArrayList<>();
        boolean closed, cancel;
        RuntimeException failure, cancellation = new java.util.concurrent.CancellationException();

        @Override public String name() { return "Frisky test"; }
        @Override public String backend() { return "CPU"; }
        @Override public ModelInspection inspection() { return info(); }
        @Override public void close() { closed = true; }

        @Override public String stream(String prompt, SamplingSettings sampling, Consumer<String> listener) {
            prompts.add(prompt);
            settings.add(sampling);
            if (failure != null) throw failure;
            listener.accept("Hello");
            if (cancel) {
                console.interrupt.run();
                throw cancellation;
            }
            listener.accept("Hello!");
            return "Hello!";
        }
    }
}
