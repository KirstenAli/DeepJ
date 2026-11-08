package io.github.kirstenali.deepj.cli;

import java.io.PrintWriter;
import java.util.Locale;
import java.util.Map;

final class CliView {
    private final CliConsole console;
    private final CliStyle style;

    CliView(CliConsole console, CliOptions options, Map<String, String> environment) {
        this.console = console;
        style = new CliStyle(options, console.interactive(), console.width(), console.colors(), environment);
    }

    PrintWriter out() { return console.out(); }
    String prompt() { return style.accent("› "); }

    void welcome(CliModel model) {
        out().print(style.banner());
        var info = model.inspection();
        String details = String.format(Locale.ROOT, "%,d parameters · %s · %,d-token context",
                info.parameters(), model.backend(), info.settings().config().maxSeqLen());
        out().println("\n  " + style.bold(CliText.label(model.name())) + "  " + style.dim(details));
        out().println("  " + style.dim("/help for commands · Ctrl+C to stop · Ctrl+D to exit"));
        out().println();
    }

    void response(String title) {
        out().println();
        out().println(style.accent(title));
    }

    void note(String text) { out().println(style.dim(CliText.safe(text))); }
    void error(String text) { out().println(style.warning(CliText.safe(text))); }

    void settings(SamplingSettings settings, boolean historyEnabled, boolean supportsThinking) {
        out().println();
        row("max-tokens", settings.maxTokens());
        row("temperature", settings.temperature());
        row("top-k", settings.topK());
        row("seed", settings.seed());
        row("repetition-penalty", settings.repetitionPenalty());
        row("context-window", settings.contextWindow().name().toLowerCase(Locale.ROOT));
        history(historyEnabled);
        if (supportsThinking) thinking(settings.thinking());
        note("\n/set <setting> <value> changes the next response.");
    }

    void history(boolean enabled) { row("history", enabled ? "on" : "off"); }
    void thinking(boolean enabled) { row("thinking", enabled ? "on" : "off"); }

    void help() {
        out().println();
        row("/settings", "Show generation settings");
        row("/set <name> <value>", "Change a generation setting");
        row("/alternatives", "Generate two alternatives to your last prompt");
        row("/theme <name>", "Choose aurora, ember or mono");
        row("/clear", "Clear the terminal");
        row("/exit", "Close the model and leave");
        note("\nEnter sends · Alt+Enter adds a line · ↑ recalls prompts · Tab completes commands");
        note("Conversation history starts off. Use /set history on to include earlier turns.");
    }

    private void row(String name, Object value) {
        out().printf("  %s %s%n", style.bold(String.format("%-22s", name)), value);
    }
}
