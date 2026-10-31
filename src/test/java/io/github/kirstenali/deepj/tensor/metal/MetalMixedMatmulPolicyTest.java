package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import static io.github.kirstenali.deepj.tensor.TensorTestStorage.lazy;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalMixedMatmulPolicyTest {
    @Test
    void largeProductsWithAnyBfloatOperandAreStaged() {
        Tensor left = lazy(DType.FLOAT32, 3, 3072, 1024);
        Tensor right = lazy(DType.BFLOAT16, 3, 1024, 64);
        assertTrue(MetalMixedMatmul.stageBfloat(left, right));
        assertTrue(MetalMixedMatmul.stageBfloat(right, lazy(DType.FLOAT32, 3, 64, 1024)));
        assertTrue(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 256, 256), lazy(DType.BFLOAT16, 256, 256)));
        assertTrue(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 1024, 1024), lazy(DType.BFLOAT16, 1024, 1024)));
        assertTrue(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 1024, 3072), lazy(DType.BFLOAT16, 3072, 768)));
        assertTrue(MetalMixedMatmul.stageBfloat(lazy(DType.FLOAT32, 1024, 16384), lazy(DType.BFLOAT16, 16384, 768)));
    }

    @Test
    void productsWhoseCastDominatesTheArithmeticStayOnTheCompactKernel() {
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 3072, 1024), lazy(DType.BFLOAT16, 1024, 64)));
        assertFalse(MetalMixedMatmul.stage(lazy(DType.FLOAT16, 3072, 1024), lazy(DType.FLOAT32, 1024, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.FLOAT32, 64, 1024), lazy(DType.BFLOAT16, 1024, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 64, 64), lazy(DType.BFLOAT16, 64, 64)));
    }

    @Test
    void largeHalfLeftOperandsAgainstFloat32AreStagedBecauseMpsOnlyMixesHalfOnTheRight() {
        Tensor half = lazy(DType.FLOAT16, 3, 1024, 64);
        Tensor full = lazy(DType.FLOAT32, 3, 64, 1024);
        assertTrue(MetalMixedMatmul.stage(half, full));
        assertFalse(MetalMixedMatmul.stageBfloat(half, full));
        assertFalse(MetalMixedMatmul.stage(full, lazy(DType.FLOAT16, 3, 1024, 64)));
        assertFalse(MetalMixedMatmul.stage(half, lazy(DType.FLOAT16, 3, 64, 1024)));
        assertFalse(MetalMixedMatmul.stage(lazy(DType.FLOAT16, 63, 64), lazy(DType.FLOAT32, 64, 64)));
    }

    @Test
    void smallOrNonBfloatProductsKeepTheExistingRoute() {
        Tensor left = lazy(DType.FLOAT32, 3072, 1024);
        assertFalse(MetalMixedMatmul.stageBfloat(left, lazy(DType.FLOAT32, 1024, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(left, lazy(DType.FLOAT16, 1024, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 63, 64), lazy(DType.BFLOAT16, 64, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 64, 64), lazy(DType.BFLOAT16, 64, 63)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 64, 63), lazy(DType.BFLOAT16, 63, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.FLOAT32, 64), lazy(DType.BFLOAT16, 64, 64)));
        assertFalse(MetalMixedMatmul.stageBfloat(lazy(DType.BFLOAT16, 64, 64), lazy(DType.BFLOAT16, 64)));
    }
}
