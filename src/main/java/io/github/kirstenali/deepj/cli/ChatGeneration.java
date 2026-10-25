package io.github.kirstenali.deepj.cli;

import java.util.concurrent.CancellationException;

final class ChatGeneration {
    private final CliConsole console;
    private final CliModel model;

    ChatGeneration(CliConsole console, CliModel model) {
        this.console = console;
        this.model = model;
    }

    boolean run(String prompt, SamplingSettings settings, CliView view, String title) {
        view.response(title);
        StreamedText output = new StreamedText(console.out());
        Thread generation = Thread.currentThread();
        try (var ignored = console.interrupt(generation::interrupt)) {
            output.complete(model.stream(prompt, settings, output::update));
            finish(output, view);
            return true;
        } catch (RuntimeException failure) {
            return stopped(failure, view);
        } finally {
            Thread.interrupted();
            view.out().println();
            view.out().flush();
        }
    }

    private boolean stopped(RuntimeException failure, CliView view) {
        if (!(failure instanceof CancellationException) && !Thread.currentThread().isInterrupted()) throw failure;
        view.out().println();
        view.note("Stopped.");
        return false;
    }

    private void finish(StreamedText output, CliView view) {
        if (!output.endsWithNewline()) view.out().println();
        if (output.empty()) view.note("No text generated. Try another seed or a different prompt.");
    }
}
