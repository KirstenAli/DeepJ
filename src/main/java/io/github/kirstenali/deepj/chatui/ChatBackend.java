package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

final class ChatBackend {
    private static DesktopBackend startup = DesktopBackend.AUTO;

    private ChatBackend() {}

    static void configure() {
        configureChoice(startup);
    }

    static void select(DesktopBackend choice) { startup = choice; }

    static void configureChoice(DesktopBackend choice) {
        boolean metal = choice != DesktopBackend.CPU && DesktopPlatform.metalAvailable();
        if (choice == DesktopBackend.METAL && !metal) {
            throw new IllegalStateException("Metal backend is not available on this computer");
        }
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
    }

    static String readyStatus() {
        return "Ready on " + name();
    }

    static String name() { return Tensor.backend() instanceof MetalBackend ? "Metal" : "CPU"; }
}
