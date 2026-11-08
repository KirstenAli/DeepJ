package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import java.util.function.Consumer;

/** Streams each agent reasoning step before approval or the final answer. */
final class CliReasoning implements Consumer<Agent.Reasoning>, AutoCloseable {
    private final CliView view;
    private StreamedText output;
    private int step = -1;
    private boolean complete;

    CliReasoning(CliView view) { this.view = view; }

    @Override public void accept(Agent.Reasoning progress) {
        if (progress.step() != step) start(progress.step());
        if (complete) return;
        output.update(progress.text());
        if (progress.complete()) {
            output.complete(progress.text());
            close();
            complete = true;
        }
    }

    private void start(int next) {
        close();
        view.response("Thinking");
        output = new StreamedText(view.out());
        step = next;
        complete = false;
    }

    @Override public void close() {
        if (output != null && !complete && !output.endsWithNewline()) view.out().println();
        view.out().flush();
    }
}
