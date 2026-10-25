package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CrossEntropyNumericalGradientTest {

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void logitsGradientMatchesFiniteDifferences() {
        Tensor logits = Tensor.from2D(new float[][]{{0.2f, -0.4f, 0.7f}, {1.1f, 0.3f, -0.2f}});
        int[] targets = {2, 0};
        Tensor analytic = CrossEntropyLoss.gradient(logits, targets);
        for (int i = 0; i < logits.rawData().length; i++) {
            assertEquals(numericalGradient(logits, targets, i), analytic.rawData()[i], 2e-4f,
                    "gradient at flat index " + i);
        }
    }

    private static float numericalGradient(Tensor logits, int[] targets, int index) {
        float original = logits.rawData()[index];
        float epsilon = 1e-3f;
        logits.rawData()[index] = original + epsilon;
        float plus = CrossEntropyLoss.loss(logits, targets);
        logits.rawData()[index] = original - epsilon;
        float minus = CrossEntropyLoss.loss(logits, targets);
        logits.rawData()[index] = original;
        return (plus - minus) / (2.0f * epsilon);
    }
}
