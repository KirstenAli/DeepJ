package io.github.kirstenali.deepj.models.nova;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertParameterGradients;

class DeepJNovaNumericalGradientTest {

    @BeforeEach
    void useCpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void allParametersIncludingSharedEmbeddingsMatchFiniteDifferences(boolean qkNorm) {
        var config = new DeepJNovaConfig(7, 3, 8, 4, 2, 2, 6, qkNorm, 0.5f, 1.0f);
        var model = new DeepJNova(config, 42L);
        int[] tokens = {1, 3, 5};
        Tensor upstream = Tensor.random(3, 7, new Random(99L));
        model.zeroGrad();
        model.forward(tokens);
        model.backward(upstream);
        assertParameterGradients(model.parameters(), () -> model.forward(tokens), upstream,
                1e-3f, 4e-3f, 7e-2f);
    }
}
