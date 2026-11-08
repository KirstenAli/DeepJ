package io.github.kirstenali.deepj.chatui;

import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DesktopBackendTest {
    @Test
    void noArgumentUsesAutoAndBothArgumentFormsAreAccepted() {
        assertEquals(DesktopBackend.AUTO, DesktopBackend.fromArguments(List.of()));
        for (DesktopBackend backend : DesktopBackend.values()) {
            assertEquals(backend, DesktopBackend.fromArguments(List.of("--backend", backend.id())));
            assertEquals(backend, DesktopBackend.fromArguments(List.of("--backend=" + backend.id())));
        }
    }

    @Test
    void invalidOrAmbiguousArgumentsAreRejectedBeforeLoadingAModel() {
        for (var arguments : List.of(List.of("--backend"), List.of("--backend="),
                List.of("--backend", "gpu"), List.of("--cpu"),
                List.of("--backend", "cpu", "--backend", "metal"))) {
            assertThrows(IllegalArgumentException.class, () -> DesktopBackend.fromArguments(arguments));
        }
    }
}
