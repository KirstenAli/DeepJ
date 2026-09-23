package io.github.kirstenali.deepj.models.prism;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertParameterGradients;

class DeepJPrismModelNumericalGradientTest {

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void fullModelParameterGradientsMatchFiniteDifferences() {
        DeepJPrismConfig config = new DeepJPrismConfig(7, 3, 4, 2, 1, 6, 3, 2);
        DeepJPrismModel model = new DeepJPrismModel(config, 42L);
        int[] inputIds = {1, 3, 5};
        Tensor upstream = Tensor.random(3, 7, new Random(99L));
        model.zeroGrad();
        model.forward(inputIds);
        model.backward(upstream);
        assertParameterGradients(model.parameters(), () -> model.forward(inputIds), upstream,
                1e-3f, 4e-3f, 7e-2f);
    }
}
