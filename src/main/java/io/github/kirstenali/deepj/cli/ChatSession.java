package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.AgentHistory;
import org.jline.reader.EndOfFileException;
import org.jline.reader.UserInterruptException;
import java.util.List;

final class ChatSession {
    private final CliConsole console;
    private final CliOptions options;
    private final CliServices services;
    private final ChatGeneration generation;
    private final CliModel model;
    private CliView view;
    private SamplingSettings settings;
    private final AgentHistory history = new AgentHistory();
    private boolean historyEnabled;
    private Request last;

    ChatSession(CliConsole console, CliModel model, CliOptions options,
                CliServices services, SamplingSettings settings) {
        this.console = console;
        this.model = model;
        this.options = options;
        this.services = services;
        this.settings = settings;
        generation = new ChatGeneration(console, model);
        view = new CliView(console, options, services.environment());
    }

    void run() {
        view.welcome(model);
        while (next()) {
            view.out().flush();
        }
    }

    private boolean next() {
        try {
            String input = console.readLine(view.prompt());
            if (input.isBlank()) return true;
            if (input.startsWith("/") && !input.startsWith("//")) return ChatCommands.run(this, input);
            generate(input.startsWith("//") ? input.substring(1) : input);
        } catch (EndOfFileException ended) {
            return false;
        } catch (UserInterruptException cancelled) {
            view.out().println();
        } catch (IllegalArgumentException | IllegalStateException failure) {
            view.error(failure.getMessage());
        }
        return true;
    }

    private void generate(String prompt) {
        List<Agent.Turn> context = historyEnabled ? history.snapshot() : List.of();
        last = new Request(prompt, settings, context);
        String reply = generation.run(prompt, context, settings, view, "DeepJ");
        if (reply != null) history.remember(Agent.Turn.answered(prompt, reply));
    }

    void alternatives() {
        if (last == null) {
            view.note("Send a prompt first, then use /alternatives.");
            return;
        }
        List<Agent.Turn> context = historyEnabled ? last.history() : List.of();
        if (generation.run(last.prompt(), context, last.settings().alternativePenalty(), view, "Alternative 1")
                != null) {
            generation.run(last.prompt(), context, last.settings().alternativeSeed(), view, "Alternative 2");
        }
    }

    void theme(CliTheme theme) {
        options.theme = theme;
        view = new CliView(console, options, services.environment());
        view.note("Theme: " + theme.name().toLowerCase(java.util.Locale.ROOT));
    }

    void edit(String name, String value) {
        if (name.equals("history")) historyEnabled = ChatCommands.historyEnabled(value);
        else settings = settings.edit(name, value);
    }

    void settings() { view.settings(settings, historyEnabled); }
    void help() { view.help(); }
    void clear() { console.clear(); }

    private record Request(String prompt, SamplingSettings settings, List<Agent.Turn> history) {}
}
