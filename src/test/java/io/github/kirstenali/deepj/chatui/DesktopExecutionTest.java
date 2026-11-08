package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DesktopExecutionTest {
    @Test
    void savedChoiceAppliesToTheNextLaunchWithoutSwitchingTheRunningBackend() {
        var value = new AtomicReference<>("auto");
        var preference = new DesktopBackendPreference(value::get, value::set);
        var settings = new DesktopExecution(preference, preference.selected());
        var current = Tensor.backend();
        assertTrue(settings.save("cpu"));
        assertEquals("cpu", settings.selected());
        assertEquals("auto", settings.startup());
        assertSame(current, Tensor.backend());
        var restarted = new DesktopExecution(preference, preference.selected());
        assertEquals("cpu", restarted.startup());
    }

    @Test
    void invalidStoredChoiceFallsBackToAutoAndInvalidInputDoesNotOverwriteIt() {
        var value = new AtomicReference<>("unknown");
        var preference = new DesktopBackendPreference(value::get, value::set);
        var settings = new DesktopExecution(preference, DesktopBackend.AUTO);
        assertEquals("auto", settings.selected());
        assertFalse(settings.save("gpu"));
        assertFalse(settings.save(null));
        assertEquals("unknown", value.get());
    }

    @Test
    void persistenceFailureIsReportedWithoutChangingTheCurrentBackend() {
        var preference = new DesktopBackendPreference(() -> "auto", value -> {
            throw new IllegalStateException("Preferences unavailable");
        });
        var settings = new DesktopExecution(preference, DesktopBackend.AUTO);
        var backend = Tensor.backend();
        assertFalse(settings.save("metal"));
        assertEquals("auto", settings.selected());
        assertSame(backend, Tensor.backend());
    }
}
