package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3RotaryTablesTest {
    @Test void conventionsNestedScopesAndForwardCallsRemainIndependent() {
        AtomicInteger builds = new AtomicInteger();
        Supplier<Tensor[]> build = () -> new Tensor[]{new Tensor(builds.incrementAndGet())};
        try (var outer = Qwen3RotaryTables.open()) {
            Tensor[] first = Qwen3RotaryTables.get(64, 10000, 5, 7, build);
            assertSame(first, Qwen3RotaryTables.get(64, 10000, 5, 7, build));
            assertNotSame(first, Qwen3RotaryTables.get("gemma2-bfloat16", 64, 10000, 5, 7, build));
            try (var inner = Qwen3RotaryTables.open()) {
                assertNotSame(first, Qwen3RotaryTables.get(64, 10000, 5, 7, build));
            }
            assertSame(first, Qwen3RotaryTables.get(64, 10000, 5, 7, build));
        }
        Qwen3RotaryTables.get(64, 10000, 5, 7, build);
        assertEquals(4, builds.get());
    }
}
