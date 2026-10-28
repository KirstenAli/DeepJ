package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.ActivationOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalTypedAttentionRoundingTest extends MetalDifferentialTestSupport {
    @Test
    void axisSoftmaxRoundsTheSameHalfMidpoint() {
        Tensor scores = Tensor.from(new float[]{-0.0087890625f, 0}, DType.FLOAT16, 2);
        assertArrayEquals(new float[]{0.497802734375f, 0.501953125f}, scores.softmax(0).toFloatArray());
    }

    @Test
    void halfProbabilitiesRoundFromFloat32AtMidpoint() {
        Tensor scores = Tensor.from(new float[]{0, 0, -0.0478515625f, -0.0302734375f}, DType.FLOAT16, 2, 2);
        float second = (float) (1.0 / (1.0 + Math.exp(-0.0087890625)));
        assertArrayEquals(new float[]{0.502197265625f}, new float[]{second});
        Tensor expected = Tensor.from(new float[]{1, 0, 1 - second, second}, DType.FLOAT16, 2, 2);
        Tensor result = ActivationOps.causalSoftmax(scores, 2, 0.5f);
        assertArrayEquals(expected.toFloatArray(), result.toFloatArray());
    }
}
