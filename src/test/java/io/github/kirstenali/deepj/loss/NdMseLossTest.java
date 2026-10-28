package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.*;

class NdMseLossTest extends NdMseLossContract {
    private TensorBackend previous;
    @BeforeEach void cpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }
    @AfterEach void restore() { Tensor.setBackend(previous); }
}
