package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.testing.NdCacheAssertions;
import org.junit.jupiter.api.Test;

class MetalNdInferenceCacheTest extends MetalDifferentialTestSupport {
    @Test
    void distinctBatchHeadsAndRepeatedRingEvictionsMatchIndependentDoublePrecisionHistory() {
        NdCacheAssertions.assertRollingBatches();
    }
}
