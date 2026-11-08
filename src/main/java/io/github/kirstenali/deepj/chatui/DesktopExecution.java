package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

/** Settings bridge; saving a preference never switches a running model's backend. */
public final class DesktopExecution {
    private final DesktopBackendPreference preference;
    private final DesktopBackend startup;

    DesktopExecution() { this(ChatBackend.preference(), ChatBackend.startupSelection()); }

    DesktopExecution(DesktopBackendPreference preference, DesktopBackend startup) {
        this.preference = preference;
        this.startup = startup;
    }

    public String selected() { return preference.selected().id(); }
    public String startup() { return startup.id(); }
    public String current() { return ChatBackend.name(); }
    public boolean metalAvailable() { return MetalBackend.isAvailable(); }

    public boolean save(String value) {
        try {
            preference.save(DesktopBackend.parse(value));
            return true;
        } catch (IllegalArgumentException | NullPointerException | IllegalStateException | SecurityException error) {
            return false;
        }
    }
}
