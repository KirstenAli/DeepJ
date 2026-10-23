package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.transformer.TransformerStack;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SharedLayerTrainingTest {
    @Test
    void repeatedLayerAccumulatesBothUsesButUpdatesItsParametersOnlyOnce() {
        Tensor.setBackend(new CpuBackend());
        Linear shared = linear();
        TransformerStack model = new TransformerStack(List.of(shared, shared));
        assertEquals(shared.parameters(), model.parameters());
        model.forward(Tensor.ones(1, 1)).backward();
        assertEquals(4, shared.weight().grad().item());
        assertEquals(3, shared.bias().grad().item());
        new AdamW(0.1f, 0.9f, 0.999f, 1e-8f, 0).step(model.parameters());
        assertEquals(1.9f, shared.weight().value.item(), 1e-5f);
        assertEquals(-0.1f, shared.bias().value.item(), 1e-5f);
    }

    private static Linear linear() {
        Linear layer = new Linear(1, 1, new Random(6));
        try (var ignored = GradMode.noGrad()) {
            layer.weight().value.set(0, 0, 2);
        }
        return layer;
    }
}
