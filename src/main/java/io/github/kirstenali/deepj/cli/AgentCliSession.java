package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.AgentHistory;
import io.github.kirstenali.deepj.agent.AgentModel;
import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import org.jline.reader.EndOfFileException;
import org.jline.reader.UserInterruptException;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CancellationException;

final class AgentCliSession {
    private final CliConsole console;
    private final CliView view;
    private final Agent agent;
    private final AgentHistory history = new AgentHistory();
    private boolean historyEnabled;

    AgentCliSession(CliConsole console, CliView view, AgentModel model, Path directory) {
        this.console = console;
        this.view = view;
        agent = new InteractiveTerminal(directory, this::approve).agent(model);
    }

    void run() {
        view.note("Agent mode · /exit to leave");
        while (next()) {
            view.out().flush();
        }
    }

    private boolean next() {
        try {
            String prompt = console.readLine(view.prompt()).strip();
            if (prompt.equals("/exit")) return false;
            if (!prompt.isEmpty()) dispatch(prompt);
        } catch (EndOfFileException ended) {
            return false;
        } catch (UserInterruptException | CancellationException | InterruptedException stopped) {
            view.note("Stopped.");
        } catch (IllegalArgumentException | IllegalStateException failure) {
            view.error(failure.getMessage());
        }
        return true;
    }

    private void dispatch(String prompt) throws InterruptedException {
        String[] words = prompt.split("\\s+");
        switch (words[0]) {
            case "/help" -> view.note("/set history on|off · /settings · /exit · Ctrl+C stops the current turn");
            case "/settings" -> view.history(historyEnabled);
            case "/set" -> setHistory(words);
            default -> answer(prompt);
        }
    }

    private void setHistory(String[] words) {
        if (words.length != 3 || !words[1].equals("history")) {
            throw new IllegalArgumentException("Use /set history on or /set history off.");
        }
        historyEnabled = ChatCommands.historyEnabled(words[2]);
        view.history(historyEnabled);
    }

    private void answer(String prompt) throws InterruptedException {
        Thread worker = Thread.currentThread();
        try (var ignored = console.interrupt(worker::interrupt)) {
            var turn = agent.runTurn(historyEnabled ? history.snapshot() : List.of(), prompt, null);
            history.remember(turn);
            view.response("DeepJ Agent");
            view.out().println(CliText.safe(turn.reply()));
        } finally {
            Thread.interrupted();
        }
    }

    private boolean approve(InteractiveTerminal.Request request) {
        view.note(request.preview());
        if (!console.interactive()) {
            view.note("Command denied. Approval requires an interactive terminal.");
            return false;
        }
        String answer = console.readLine("Run once? [y/N] ").strip();
        return answer.equalsIgnoreCase("y") || answer.equalsIgnoreCase("yes");
    }
}
