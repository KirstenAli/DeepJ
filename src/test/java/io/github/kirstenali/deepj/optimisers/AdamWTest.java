package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

public class AdamWTest {

    @Test
    void step_matchesKnown1x1Case_withoutWeightDecay() {
        AdamW opt = new AdamW(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

        Parameter p = new Parameter(rowTensor(1.0f));
        p.grad = rowTensor(1.0f);

        opt.step(List.of(p));
        Assertions.assertEquals(0.9f, p.value.data[0], 1e-6f);

        opt.step(List.of(p));
        Assertions.assertEquals(0.8f, p.value.data[0], 1e-6f);
    }

    @Test
    void weightDecay_shrinksWeightsEvenWithZeroGrad() {
        AdamW opt = new AdamW(0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);

        Parameter p = new Parameter(rowTensor(10.0f));
        p.grad = rowTensor(0.0f);

        opt.step(List.of(p));
        Assertions.assertTrue(p.value.data[0] < 10.0f);
    }

    @Test
    void rejectsInvalidHyperparams() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.0f, 0.9f, 0.999f, 1e-8f, 0.0f));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1f, 1.0f, 0.999f, 1e-8f, 0.0f));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1f, 0.9f, 0.0f, 1e-8f, 0.0f));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1f, 0.9f, 0.999f, 0.0f, 0.0f));
    }

    private static Tensor rowTensor(float... values) {
        return TensorAdapters.unpackF32(values, 1, values.length);
    }
}
