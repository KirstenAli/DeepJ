package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.testing.PackedLayoutAssertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalPackedLayoutTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void compactAndStridedHeadsProductsCacheWritesAndAliasSnapshotsUseGpuLayouts(DType dtype) {
        PackedLayoutAssertions.heads(metal, dtype);
        for (boolean ta : new boolean[]{false, true}) {
            for (boolean tb : new boolean[]{false, true}) {
                PackedLayoutAssertions.batchedProducts(metal, dtype, ta, tb);
            }
        }
        Tensor cache = PackedLayoutAssertions.cache(metal, dtype);
        assertFalse(TensorTestStorage.hasHostStorage(cache));
        PackedLayoutAssertions.cacheAliasing(metal, dtype);
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void fusedCapabilityNeverPromisesUnsupportedCompactStorage(DType dtype) {
        Tensor query = new Tensor(dtype, 256, 64);
        assertEquals(dtype == DType.FLOAT32 && MetalNative.TILED_ATTENTION_AVAILABLE,
                metal.supportsCausalAttention(query, query, query, 1, 0.125f));
        if (dtype != DType.FLOAT32) {
            assertThrows(UnsupportedOperationException.class,
                    () -> metal.causalAttention(query, query, query, 1, 0.125f));
        }
    }
}
