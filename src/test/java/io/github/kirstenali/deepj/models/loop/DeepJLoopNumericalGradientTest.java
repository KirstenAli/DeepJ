package io.github.kirstenali.deepj.models.loop;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertParameterGradients;

class DeepJLoopNumericalGradientTest {

    @BeforeEach
    void selectCpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @CsvSource({"1,false", "2,false", "3,false", "1,true", "2,true", "3,true"})
    void allWeightsMatchFiniteDifferencesThroughEveryPass(int passes, boolean normalize) {
        var settings = new DeepJLoopConfig(7, 3, 4, 2, 1, 0, 1, 0, passes, 6,
                normalize, 0.5f, 1.0f, 0.5f);
        var decoder = new DeepJLoop(settings, 19L);
        int[] sequence = {1, 3, 5};
        Tensor upstream = Tensor.random(3, 7, new Random(29L));
        decoder.zeroGrad();
        decoder.forward(sequence);
        decoder.forward(sequence).backward(upstream);
        assertParameterGradients(decoder.parameters(), () -> decoder.forward(sequence), upstream,
                1e-3f, 3e-3f, 6e-2f);
    }
}
