package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.PackedLayoutAssertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CpuPackedLayoutTest {
    private final CpuBackend backend = new CpuBackend();
    @BeforeEach void cpu() { Tensor.setBackend(backend); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void compactAndStridedHeadsProductsCacheWritesAndAliasSnapshotsUseNdStorage(DType dtype) {
        PackedLayoutAssertions.heads(backend, dtype);
        for (boolean ta : new boolean[]{false, true}) {
            for (boolean tb : new boolean[]{false, true}) {
                PackedLayoutAssertions.batchedProducts(backend, dtype, ta, tb);
            }
        }
        PackedLayoutAssertions.cache(backend, dtype);
        PackedLayoutAssertions.cacheAliasing(backend, dtype);
    }

    @Test
    void cacheWritesRejectSavedAliasesAndBroadcastDestinations() {
        Tensor parent = new Tensor(8, 2).requiresGrad(true);
        Tensor loss = parent.multiply(parent).sumTensor(), alias = backend.view(parent, 8, 2);
        assertThrows(IllegalArgumentException.class, () -> backend.writeHeadCache(alias, new Tensor(2, 2), 2, 0));
        Tensor repeated = new Tensor(1, 2).expand(8, 2);
        assertThrows(IllegalStateException.class, () -> backend.writeHeadCache(repeated, new Tensor(2, 2), 2, 0));
        assertThrows(IllegalArgumentException.class, () -> backend.splitHeads(new Tensor(DType.FLOAT16, 3, 4), 3));
        assertThrows(IllegalArgumentException.class, () -> backend.mergeHeads(new Tensor(DType.FLOAT16, 3, 4), 2));
        assertNotNull(loss);
    }
}
