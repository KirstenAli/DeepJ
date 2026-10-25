package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import java.util.function.BooleanSupplier;

enum BackendChoice {
    AUTO, CPU, METAL;

    BackendChoice resolve(BooleanSupplier available) {
        if (this == CPU) return CPU;
        if (available.getAsBoolean()) return METAL;
        if (this == METAL) throw new IllegalArgumentException("Metal is unavailable. Use --backend cpu or auto.");
        return CPU;
    }

    void activate() {
        BackendChoice selected = resolve(MetalBackend::isAvailable);
        Tensor.setBackend(selected == METAL ? new MetalBackend() : new CpuBackend());
    }
}
