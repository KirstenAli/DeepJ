package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

abstract class NeuralTest {
    @BeforeEach void selectCpu() { Tensor.setBackend(new CpuBackend()); }
    @AfterEach void releaseBackend() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }
}
