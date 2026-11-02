package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.UnaryOperator;

import static org.junit.jupiter.api.Assertions.*;

class RecomputeTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void gradientsMatchTheSavedGraphExactly() {
        Tensor weightA = weight(), inputA = input(), weightB = weight(), inputB = input();
        block(weightA).apply(inputA).backward(ones());
        Recompute.forward(block(weightB), inputB).backward(ones());
        assertArrayEquals(weightA.grad().rawData(), weightB.grad().rawData());
        assertArrayEquals(inputA.grad().rawData(), inputB.grad().rawData());
    }

    @Test
    void blockRunsOnceInForwardAndOnceMoreInBackward() {
        AtomicInteger calls = new AtomicInteger();
        Tensor weight = weight();
        UnaryOperator<Tensor> counted = counting(calls, block(weight));
        Tensor output = Recompute.forward(counted, input());
        assertEquals(1, calls.get());
        output.backward(ones());
        assertEquals(2, calls.get());
        assertNotNull(weight.grad());
    }

    @Test
    void untrackedInputsBypassRecomputeAndRestoreGradMode() {
        AtomicInteger calls = new AtomicInteger();
        UnaryOperator<Tensor> counted = counting(calls, value -> value.multiplyScalar(2));
        try (var ignored = GradMode.noGrad()) {
            assertFalse(Recompute.forward(counted, input()).requiresGrad());
            assertFalse(GradMode.isEnabled());
        }
        assertEquals(1, calls.get());
        assertTrue(GradMode.isEnabled());
    }

    private static UnaryOperator<Tensor> counting(AtomicInteger calls, UnaryOperator<Tensor> block) {
        return value -> {
            calls.incrementAndGet();
            return block.apply(value);
        };
    }

    private static UnaryOperator<Tensor> block(Tensor weight) {
        return value -> value.matmul(weight).tanhActivation().matmul(weight);
    }

    private static Tensor weight() {
        return Tensor.from(new float[]{0.5f, -0.25f, 0.75f, 0.1f}, 2, 2).requiresGrad(true);
    }

    private static Tensor input() {
        return Tensor.from(new float[]{1, 2, 3, 4}, 2, 2).requiresGrad(true);
    }

    private static Tensor ones() {
        return Tensor.from(new float[]{1, 1, 1, 1}, 2, 2);
    }
}
