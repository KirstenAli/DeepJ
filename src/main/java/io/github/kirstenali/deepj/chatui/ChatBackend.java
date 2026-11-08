package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

final class ChatBackend {
    private static final DesktopBackendPreference PREFERENCE = DesktopBackendPreference.local();
    private static final DesktopBackend STARTUP = PREFERENCE.selected();

    private ChatBackend() {}

    static void configure() {
        configureChoice(STARTUP);
    }

    static void configureChoice(DesktopBackend choice) {
        boolean metal = choice != DesktopBackend.CPU && DesktopPlatform.metalAvailable();
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
    }

    static String readyStatus() {
        return "Ready on " + name();
    }

    static String name() { return Tensor.backend() instanceof MetalBackend ? "Metal" : "CPU"; }
    static DesktopBackendPreference preference() { return PREFERENCE; }
    static DesktopBackend startupSelection() { return STARTUP; }
}
