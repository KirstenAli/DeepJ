package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

final class ChatBackend {

    private ChatBackend() {}

    static void configure() {
        Tensor.setBackend(MetalBackend.isAvailable() ? new MetalBackend() : new CpuBackend());
    }

    static String readyStatus() {
        return "Ready on " + (Tensor.backend() instanceof MetalBackend ? "Metal" : "CPU");
    }
}
