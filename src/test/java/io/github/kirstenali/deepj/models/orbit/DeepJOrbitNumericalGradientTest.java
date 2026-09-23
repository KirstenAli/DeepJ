package io.github.kirstenali.deepj.models.orbit;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertParameterGradients;

class DeepJOrbitNumericalGradientTest {

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void fullModelParameterGradientsMatchFiniteDifferences() {
        DeepJOrbit model = new DeepJOrbit(new DeepJOrbitConfig(7, 3, 4, 2, 1, 6), 42L);
        int[] inputIds = {1, 3, 5};
        Tensor upstream = Tensor.random(3, 7, new Random(99L));
        model.zeroGrad();
        model.forward(inputIds);
        model.backward(upstream);
        assertParameterGradients(model.parameters(), () -> model.forward(inputIds), upstream,
                1e-3f, 4e-3f, 7e-2f);
    }
}
