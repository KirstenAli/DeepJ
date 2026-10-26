package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.NdCacheAssertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class NdInferenceContextTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void distinctBatchHeadsAndRingEvictionsMatchIndependentDoublePrecisionHistory() {
        NdCacheAssertions.assertRollingBatches();
    }

    @Test
    void invalidIncrementalAxesOwnersAndDtypesAreRejectedBeforeCacheWrites() {
        Object owner = new Object();
        Tensor q = new Tensor(2, 4, 1, 2), kv = new Tensor(2, 2, 1, 2);
        try (var ignored = GradMode.noGrad(); var context = new InferenceContext(4)) {
            context.begin(0);
            context.attend(owner, q, kv, kv, 4, 2, 0.5f);
            context.finish();
            context.begin(1);
            assertThrows(IllegalStateException.class, () -> context.attend(new Object(), q, kv, kv, 4, 2, 0.5f));
            context.begin(1);
            assertThrows(IllegalArgumentException.class,
                    () -> context.attend(owner, q, kv.to(DType.FLOAT16), kv, 4, 2, 0.5f));
            context.begin(1);
            assertThrows(IllegalArgumentException.class,
                    () -> context.attend(owner, new Tensor(1, 4, 1, 2), kv, kv, 4, 2, 0.5f));
        }
    }

    @Test
    void cacheInputValidationCoversRanksHeadGroupsEmptyAxesAndTimeMismatch() {
        Tensor q = new Tensor(2, 4, 1, 2), kv = new Tensor(2, 2, 1, 2);
        assertThrows(IllegalArgumentException.class, () -> CacheInputs.of(q, kv, kv, 3, 2));
        assertThrows(IllegalArgumentException.class, () -> CacheInputs.of(q, kv, kv, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> CacheInputs.of(q, kv, kv, 4, 0));
        assertThrows(IllegalArgumentException.class, () -> CacheInputs.of(new Tensor(3, 2), kv, kv, 4, 2));
        assertThrows(IllegalArgumentException.class,
                () -> CacheInputs.of(new Tensor(2, 4, 2, 2), kv, kv, 4, 2));
        assertThrows(IllegalArgumentException.class,
                () -> CacheInputs.of(new Tensor(0, 4, 1, 2), new Tensor(0, 2, 1, 2), new Tensor(0, 2, 1, 2), 4, 2));
    }
}
