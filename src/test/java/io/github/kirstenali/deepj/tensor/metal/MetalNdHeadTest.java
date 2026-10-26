package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdHeadTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void typedHeadCoordinatesRoundTripOnGpuWithoutHostShadows(DType dtype) {
        Tensor input = metal.copy(Tensor.from(values(48), dtype, 2, 3, 8)).requiresGrad(true);
        Tensor heads = LayoutOps.splitHeads(input, 4), restored = LayoutOps.mergeHeads(heads, 4);
        assertArrayEquals(new int[]{2, 4, 3, 2}, heads.shape());
        restored.sumTensor().backward();
        assertFalse(TensorTestStorage.hasHostStorage(input));
        assertFalse(TensorTestStorage.hasHostStorage(heads));
        assertArrayEquals(input.toFloatArray(), restored.toFloatArray());
        assertEquals(DType.FLOAT32, input.grad().dtype());
        for (float derivative : input.grad().toFloatArray()) { assertEquals(1, derivative); }
    }

    @Test
    void ndGroupedFlashUsesExactExistingPackedKernelWithBatchIsolatedHeads() {
        Tensor query = Tensor.from(values(2 * 4 * 256 * 64), 2, 4, 256, 64).requiresGrad(true);
        Tensor key = Tensor.from(values(2 * 2 * 256 * 64), 2, 2, 256, 64).requiresGrad(true);
        Tensor value = key.detach().addScalar(0.1f).requiresGrad(true);
        Tensor actual = AttentionOps.causal(query, key, value, 4, 2, 0.125f);
        actual.sumTensor().backward();
        Tensor[] gradients = {query.grad().detach(), key.grad().detach(), value.grad().detach()};
        Tensor q = HeadLayout.pack(query.detach()).requiresGrad(true);
        Tensor k = HeadLayout.pack(key.detach()).requiresGrad(true);
        Tensor v = HeadLayout.pack(value.detach()).requiresGrad(true);
        Tensor packed = AttentionOps.causal(q, k, v, 8, 4, 0.125f);
        packed.sumTensor().backward();
        assertArrayEquals(packed.toFloatArray(), actual.toFloatArray());
        assertArrayEquals(q.grad().toFloatArray(), gradients[0].toFloatArray());
        assertArrayEquals(k.grad().toFloatArray(), gradients[1].toFloatArray());
        assertArrayEquals(v.grad().toFloatArray(), gradients[2].toFloatArray());
    }

    @Test
    void ndAttentionAndRotaryHaveActualGpuFiniteDifferenceAdjoints() {
        Tensor query = Tensor.from(values(8), 2, 2, 2, 1);
        Tensor key = Tensor.from(values(4), 2, 1, 2, 1), value = key.detach().addScalar(0.2f);
        Tensor upstream = query.detach().addScalar(0.2f);
        assertTensorGradients(query, q -> AttentionOps.causal(q, key, value, 2, 1, 0.5f),
                upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(key, k -> AttentionOps.causal(query, k, value, 2, 1, 0.5f),
                upstream, 1e-3f, 3e-4f, 2e-3f);
        assertTensorGradients(value, v -> AttentionOps.causal(query, key, v, 2, 1, 0.5f),
                upstream, 1e-3f, 3e-4f, 2e-3f);
        RotaryEmbedding rotary = new RotaryEmbedding(4, 8);
        Tensor positions = Tensor.from(values(48), 2, 2, 3, 4);
        assertTensorGradients(positions, x -> rotary.apply(x, 3, 2), positions.detach(),
                1e-3f, 3e-4f, 2e-3f);
    }

    private static float[] values(int count) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = (index % 19 - 9) * 0.03f; }
        return values;
    }
}
