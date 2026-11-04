package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import java.util.List;
import java.util.concurrent.CancellationException;

final class ChatGeneration {
    private final CliConsole console;
    private final CliModel model;

    ChatGeneration(CliConsole console, CliModel model) {
        this.console = console;
        this.model = model;
    }

    String run(String prompt, List<Agent.Turn> history, SamplingSettings settings, CliView view, String title) {
        view.response(title);
        StreamedText output = new StreamedText(console.out());
        Thread generation = Thread.currentThread();
        try (var ignored = console.interrupt(generation::interrupt)) {
            String reply = model.streamConversation(prompt, history, settings, output::update);
            output.complete(reply);
            finish(output, view);
            return reply;
        } catch (RuntimeException failure) {
            stopped(failure, view);
            return null;
        } finally {
            Thread.interrupted();
            view.out().println();
            view.out().flush();
        }
    }

    private void stopped(RuntimeException failure, CliView view) {
        if (!(failure instanceof CancellationException) && !Thread.currentThread().isInterrupted()) throw failure;
        view.out().println();
        view.note("Stopped.");
    }

    private void finish(StreamedText output, CliView view) {
        if (!output.endsWithNewline()) view.out().println();
        if (output.empty()) view.note("No text generated. Try another seed or a different prompt.");
    }
}
