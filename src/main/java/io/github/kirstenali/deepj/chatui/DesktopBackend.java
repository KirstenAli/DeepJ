package io.github.kirstenali.deepj.chatui;

import java.util.Locale;

enum DesktopBackend {
    AUTO, CPU, METAL;

    String id() { return name().toLowerCase(Locale.ROOT); }

    static DesktopBackend parse(String value) {
        return valueOf(value.toUpperCase(Locale.ROOT));
    }
}
