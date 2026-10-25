package io.github.kirstenali.deepj.cli;

import picocli.CommandLine.Option;
import picocli.CommandLine.ScopeType;

final class CliOptions {
    enum Color { AUTO, ALWAYS, NEVER }

    @Option(names = "--backend", scope = ScopeType.INHERIT,
            description = "Compute backend: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE}).")
    BackendChoice backend = BackendChoice.AUTO;

    @Option(names = "--theme", scope = ScopeType.INHERIT,
            description = "Terminal palette: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE}).")
    CliTheme theme = CliTheme.AURORA;

    @Option(names = "--color", scope = ScopeType.INHERIT,
            description = "Color output: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE}).")
    Color color = Color.AUTO;

    @Option(names = "--no-banner", scope = ScopeType.INHERIT, description = "Hide the startup wordmark.")
    boolean noBanner;

    picocli.CommandLine.Help.Ansi helpAnsi(CliServices services) {
        if (color == Color.ALWAYS) return picocli.CommandLine.Help.Ansi.ON;
        boolean disabled = color == Color.NEVER || !services.interactive()
                || services.environment().containsKey("NO_COLOR");
        return disabled ? picocli.CommandLine.Help.Ansi.OFF : picocli.CommandLine.Help.Ansi.AUTO;
    }
}
