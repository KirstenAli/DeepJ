package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

public class AdamWTest {

    @Test
    void step_matchesKnown1x1Case_withoutWeightDecay() {
        AdamW opt = new AdamW(0.1, 0.9, 0.999, 1e-8, 0.0);

        Parameter p = new Parameter(Tensor.from2D(new double[][]{{1.0}}));
        p.grad = Tensor.from2D(new double[][]{{1.0}});

        opt.step(List.of(p));
        Assertions.assertEquals(0.9, p.value.data[0], 1e-6);

        opt.step(List.of(p));
        Assertions.assertEquals(0.8, p.value.data[0], 1e-6);
    }

    @Test
    void weightDecay_shrinksWeightsEvenWithZeroGrad() {
        AdamW opt = new AdamW(0.1, 0.9, 0.999, 1e-8, 0.1);

        Parameter p = new Parameter(Tensor.from2D(new double[][]{{10.0}}));
        p.grad = Tensor.from2D(new double[][]{{0.0}});

        opt.step(List.of(p));
        Assertions.assertTrue(p.value.data[0] < 10.0);
    }

    @Test
    void rejectsInvalidHyperparams() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0, 0.9, 0.999, 1e-8, 0));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1, 1.0, 0.999, 1e-8, 0));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1, 0.9, 0.0, 1e-8, 0));
        Assertions.assertThrows(IllegalArgumentException.class, () -> new AdamW(0.1, 0.9, 0.999, 0.0, 0));
    }
}
