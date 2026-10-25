package io.github.kirstenali.deepj.cli;

import java.util.Map;

final class CliStyle {
    private static final String[] LOGO = {
            "██████╗ ███████╗███████╗██████╗      ██╗",
            "██╔══██╗██╔════╝██╔════╝██╔══██╗     ██║",
            "██║  ██║█████╗  █████╗  ██████╔╝     ██║",
            "██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██   ██║",
            "██████╔╝███████╗███████╗██║     ╚█████╔╝",
            "╚═════╝ ╚══════╝╚══════╝╚═╝      ╚════╝"
    };
    private final CliOptions options;
    private final boolean interactive, colored;
    private final int width, colors;

    CliStyle(CliOptions options, boolean interactive, int width, int colors, Map<String, String> environment) {
        this.options = options;
        this.interactive = interactive;
        this.width = width;
        this.colors = colors;
        colored = options.color == CliOptions.Color.ALWAYS || options.color == CliOptions.Color.AUTO
                && interactive && !environment.containsKey("NO_COLOR");
    }

    String banner() {
        if (options.noBanner) return "";
        if (!interactive || width < 50) return accent("DEEPJ") + "  " + dim("Language models, built in Java.") + "\n";
        StringBuilder result = new StringBuilder("\n");
        for (int line = 0; line < LOGO.length; line++) {
            result.append("  ").append(paint(LOGO[line], options.theme.gradient[line])).append('\n');
        }
        return result + "\n  " + dim("Language models, built in Java.") + "\n";
    }

    String accent(String value) { return paint(value, options.theme.accent); }
    String bold(String value) { return effect(value, "1"); }
    String dim(String value) { return effect(value, "2"); }
    String warning(String value) { return paint(value, 215); }

    private String paint(String value, int color) {
        if (color < 0) return bold(value);
        String code = colors >= 256 ? "38;5;" + color : color >= 100 ? "35" : "36";
        return effect(value, code);
    }

    private String effect(String value, String code) {
        return colored ? "\u001b[" + code + "m" + value + "\u001b[0m" : value;
    }
}
