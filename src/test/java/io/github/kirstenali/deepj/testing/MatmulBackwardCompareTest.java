package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MatmulBackwardCompareTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void comparesEquivalentInputAndWeightGradients(boolean weightGradient) {
        var original = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
        try {
            Random random = new Random(31);
            Tensor a = Tensor.random(4, weightGradient ? 2 : 3, random);
            Tensor b = Tensor.random(weightGradient ? 4 : 2, 3, random);
            Tensor explicit = MatmulBackwardCompare.product(a, b, weightGradient, false);
            Tensor flagged = MatmulBackwardCompare.product(a, b, weightGradient, true);
            assertArrayEquals(explicit.rawData(), flagged.rawData(), 1e-6f);
        } finally {
            Tensor.setBackend(original);
        }
    }
}
