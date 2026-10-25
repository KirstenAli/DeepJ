package io.github.kirstenali.deepj.cli;

import org.jline.reader.EndOfFileException;
import org.jline.reader.UserInterruptException;

final class ChatSession {
    private final CliConsole console;
    private final CliOptions options;
    private final CliServices services;
    private final ChatGeneration generation;
    private final CliModel model;
    private CliView view;
    private SamplingSettings settings, previous;
    private String lastPrompt;

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
        lastPrompt = prompt;
        previous = settings;
        generation.run(prompt, settings, view, "DeepJ");
    }

    void alternatives() {
        if (lastPrompt == null) {
            view.note("Send a prompt first, then use /alternatives.");
            return;
        }
        if (generation.run(lastPrompt, previous.alternativePenalty(), view, "Alternative 1")) {
            generation.run(lastPrompt, previous.alternativeSeed(), view, "Alternative 2");
        }
    }

    void theme(CliTheme theme) {
        options.theme = theme;
        view = new CliView(console, options, services.environment());
        view.note("Theme: " + theme.name().toLowerCase(java.util.Locale.ROOT));
    }

    void edit(String name, String value) { settings = settings.edit(name, value); }
    void settings() { view.settings(settings); }
    void help() { view.help(); }
    void clear() { console.clear(); }
}
