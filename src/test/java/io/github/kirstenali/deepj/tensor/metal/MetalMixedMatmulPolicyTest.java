package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import static io.github.kirstenali.deepj.tensor.TensorTestStorage.lazy;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalMixedMatmulPolicyTest {
    @Test
    void onlyLargeNarrowFloat32ByBfloatProductsAreWidened() {
        Tensor left = lazy(DType.FLOAT32, 3, 3072, 1024);
        Tensor right = lazy(DType.BFLOAT16, 3, 1024, 64);
        assertTrue(MetalMixedMatmul.widenRight(left, right));
        assertTrue(MetalMixedMatmul.widenRight(lazy(DType.FLOAT32, 3, 1024, 3072),
                lazy(DType.BFLOAT16, 3, 3072, 64)));
        assertTrue(MetalMixedMatmul.widenRight(lazy(DType.FLOAT32, 3, 3072, 64),
                lazy(DType.BFLOAT16, 3, 64, 1024)));
        assertFalse(MetalMixedMatmul.widenRight(left, lazy(DType.FLOAT32, right.shape())));
        assertFalse(MetalMixedMatmul.widenRight(left, lazy(DType.FLOAT16, right.shape())));
        assertFalse(MetalMixedMatmul.widenRight(right, left));
        assertFalse(MetalMixedMatmul.widenRight(lazy(DType.FLOAT32, 64, 1024), right));
        assertFalse(MetalMixedMatmul.widenRight(left, lazy(DType.BFLOAT16, 1024, 256)));
        assertFalse(MetalMixedMatmul.widenRight(lazy(DType.FLOAT32, 64), right));
        assertFalse(MetalMixedMatmul.widenRight(left, lazy(DType.BFLOAT16, 64)));
    }
}
