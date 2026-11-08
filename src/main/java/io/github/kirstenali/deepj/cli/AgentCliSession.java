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
import java.util.function.Function;

final class AgentCliSession {
    private final CliConsole console;
    private final CliView view;
    private final InteractiveTerminal terminal;
    private final Function<Boolean, AgentModel> models;
    private Agent agent;
    private final AgentHistory history = new AgentHistory();
    private boolean historyEnabled;
    private boolean thinking;

    AgentCliSession(CliConsole console, CliView view, AgentModel model, Path directory) {
        this(console, view, directory, enabled -> plainModel(model, enabled), false);
    }

    AgentCliSession(CliConsole console, CliView view, Path directory,
                    Function<Boolean, AgentModel> models, boolean thinking) {
        this.console = console;
        this.view = view;
        this.models = models;
        terminal = new InteractiveTerminal(directory, this::approve);
        setThinking(thinking);
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
            case "/help" -> view.note("/set history on|off · /set thinking on|off · /settings · /exit · Ctrl+C stops");
            case "/settings" -> settings();
            case "/set" -> setSetting(words);
            default -> answer(prompt);
        }
    }

    private void setSetting(String[] words) {
        if (words.length != 3) throw new IllegalArgumentException("Use /set history|thinking on|off.");
        boolean enabled = ChatCommands.toggleEnabled(words[1], words[2]);
        switch (words[1]) {
            case "history" -> historyEnabled = enabled;
            case "thinking" -> setThinking(enabled);
            default -> throw new IllegalArgumentException("Use /set history|thinking on|off.");
        }
        settings();
    }

    private void settings() {
        view.history(historyEnabled);
        view.thinking(thinking);
    }

    private void setThinking(boolean enabled) {
        Agent candidate = terminal.agent(models.apply(enabled));
        agent = candidate;
        thinking = enabled;
    }

    private void answer(String prompt) throws InterruptedException {
        Thread worker = Thread.currentThread();
        try (var ignored = console.interrupt(worker::interrupt)) {
            var turn = runTurn(prompt);
            history.remember(turn);
            view.response("DeepJ Agent");
            view.out().println(CliText.safe(turn.reply()));
        } finally {
            Thread.interrupted();
        }
    }

    private Agent.Turn runTurn(String prompt) throws InterruptedException {
        var context = historyEnabled ? history.snapshot() : List.<Agent.Turn>of();
        if (!thinking) return agent.runTurn(context, prompt, null);
        try (var reasoning = new CliReasoning(view)) {
            return agent.runTurn(context, prompt, null, reasoning);
        }
    }

    private static AgentModel plainModel(AgentModel model, boolean thinking) {
        if (thinking) throw new IllegalArgumentException("Thinking is unavailable for this model.");
        return model;
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
