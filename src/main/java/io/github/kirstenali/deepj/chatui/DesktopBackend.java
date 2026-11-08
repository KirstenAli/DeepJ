package io.github.kirstenali.deepj.chatui;

import java.util.Locale;
import java.util.List;

enum DesktopBackend {
    AUTO, CPU, METAL;

    String id() { return name().toLowerCase(Locale.ROOT); }

    static DesktopBackend parse(String value) {
        try { return valueOf(value.toUpperCase(Locale.ROOT)); }
        catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("Unknown backend '" + value + "'. Choose auto, cpu, or metal.", error);
        }
    }

    static DesktopBackend fromArguments(List<String> arguments) {
        if (arguments.isEmpty()) return AUTO;
        if (arguments.size() == 1 && arguments.get(0).startsWith("--backend=")) {
            return parse(arguments.get(0).substring("--backend=".length()));
        }
        if (arguments.size() == 2 && "--backend".equals(arguments.get(0))) {
            return parse(arguments.get(1));
        }
        throw new IllegalArgumentException("Use --backend auto, --backend cpu, or --backend metal");
    }
}
