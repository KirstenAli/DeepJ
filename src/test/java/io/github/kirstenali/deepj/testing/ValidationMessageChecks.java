package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Original matrix diagnostics also identify the invalid input for vectors, batches and compact storage. */
public final class ValidationMessageChecks {
    private ValidationMessageChecks() {}

    public static void assertRejections(TensorBackend backend, DType dtype) {
        for (int[] shape : List.of(new int[]{4}, new int[]{2, 4}, new int[]{2, 1, 4})) {
            Tensor input = new Tensor(dtype, shape);
            int[] statistics = shape.clone();
            statistics[statistics.length - 1] = 2;
            Tensor badStd = new Tensor(dtype, statistics);
            assertMessage("layerNormBackward: std", () -> backend.layerNormBackward(input, input, badStd, 4));
            assertMessage("targets length", () -> backend.crossEntropyGradient(input, new int[input.numel() / 4 + 1]));
            assertNull(input.getGpuTag());
            assertNull(badStd.getGpuTag());
            assertEquals(0, input.gradientState().version());
        }
    }

    private static void assertMessage(String expected, Runnable operation) {
        var failure = assertThrows(IllegalArgumentException.class, operation::run);
        assertTrue(failure.getMessage().contains(expected), failure.getMessage());
    }
}
