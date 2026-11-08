package io.github.kirstenali.deepj.chatui;

import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

final class DesktopBackendPreference {
    private final Supplier<String> read;
    private final Consumer<String> write;

    DesktopBackendPreference(Supplier<String> read, Consumer<String> write) {
        this.read = read;
        this.write = write;
    }

    static DesktopBackendPreference local() {
        Preferences preferences = Preferences.userNodeForPackage(ChatBackend.class);
        return new DesktopBackendPreference(() -> preferences.get("backend", "auto"),
                value -> persist(preferences, value));
    }

    DesktopBackend selected() {
        try { return DesktopBackend.parse(read.get()); }
        catch (IllegalArgumentException | NullPointerException error) { return DesktopBackend.AUTO; }
    }

    void save(DesktopBackend backend) { write.accept(backend.id()); }

    private static void persist(Preferences preferences, String value) {
        String previous = preferences.get("backend", "auto");
        preferences.put("backend", value);
        try { preferences.flush(); }
        catch (BackingStoreException error) {
            preferences.put("backend", previous);
            throw new IllegalStateException("Could not save the execution preference", error);
        }
    }
}
