package io.github.kirstenali.deepj.tensor.cpu;

import io.github.kirstenali.deepj.tensor.CrossEntropyDefaults;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CpuFusedCrossEntropyTest {
    @Test void fusedLossAndGradientMatchSeparateEvaluationExactly() {
        CpuBackend backend = new CpuBackend();
        Tensor.setBackend(backend);
        for (int width : new int[]{1, 7, 256, 257, 4096}) {
            check(backend, values(9, width), new int[]{0, 0, 0, 0, 0, 0, 0, 0, width - 1});
        }
    }

    @Test void exceptionalValuesRetainExistingLossAndGradientBehavior() {
        CpuBackend backend = new CpuBackend();
        Tensor.setBackend(backend);
        check(backend, Tensor.from(new float[]{Float.NaN, 1, 2, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, -0.0f, 0.0f, -1000}, 4, 2), new int[]{0, 1, 0, 1});
    }

    private static void check(CpuBackend backend, Tensor logits, int[] targets) {
        boolean[] mask = new boolean[targets.length];
        for (int row = 0; row < mask.length; row++) { mask[row] = row % 2 == 0; }
        for (boolean[] selected : new boolean[][]{null, mask}) {
            var expected = CrossEntropyDefaults.apply(backend, logits, targets, selected);
            var actual = selected == null ? backend.crossEntropy(logits, targets)
                    : backend.crossEntropy(logits, targets, selected);
            assertEquals(expected.loss().item(), actual.loss().item());
            assertArrayEquals(expected.gradient().toFloatArray(), actual.gradient().toFloatArray());
        }
    }

    @Test void invalidInputsFailBeforeEvaluation() {
        CpuBackend backend = new CpuBackend();
        Tensor.setBackend(backend);
        Tensor logits = new Tensor(2, 3);
        assertThrows(IllegalArgumentException.class, () -> backend.crossEntropy(logits, new int[]{0}));
        assertThrows(IllegalArgumentException.class, () -> backend.crossEntropy(logits, new int[]{0, 3}));
        assertThrows(IllegalArgumentException.class,
                () -> backend.crossEntropy(logits, new int[]{0, 1}, new boolean[2]));
    }

    private static Tensor values(int rows, int width) {
        var random = new Random(37L * width);
        float[] values = new float[rows * width];
        for (int index = 0; index < values.length; index++) { values[index] = random.nextFloat() * 40 - 20; }
        return Tensor.from(values, rows, width);
    }
}
