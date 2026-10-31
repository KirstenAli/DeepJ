package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdHeadLayoutTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void splitMergeKeepBatchAndHeadCoordinatesAndAdjointsInEveryDtype() {
        float[] values = new float[48];
        for (int index = 0; index < values.length; index++) { values[index] = index; }
        for (DType dtype : DType.values()) {
            Tensor input = Tensor.from(values, dtype, 2, 3, 8).requiresGrad(true);
            Tensor heads = LayoutOps.splitHeads(input, 4);
            assertArrayEquals(new int[]{2, 4, 3, 2}, heads.shape());
            assertEquals(input.get(1, 2, 5), heads.get(1, 2, 2, 1));
            Tensor restored = LayoutOps.mergeHeads(heads, 4);
            assertArrayEquals(values, restored.toFloatArray());
            restored.sumTensor().backward();
            assertEquals(DType.FLOAT32, input.grad().dtype());
            for (float derivative : input.grad().toFloatArray()) { assertEquals(1, derivative); }
        }
    }

    @Test
    void compactGroupedNdAttentionHasActualQueryKeyAndValueFiniteDifferenceGradients() {
        Tensor q = Tensor.from(new float[]{0.1f, 0.2f, -0.1f, 0.3f, 0.4f, -0.2f, 0.2f, 0.1f}, 2, 2, 2, 1);
        Tensor k = Tensor.from(new float[]{0.3f, -0.2f, 0.1f, 0.4f}, 2, 1, 2, 1);
        Tensor v = Tensor.from(new float[]{0.4f, 0.2f, -0.2f, 0.3f}, 2, 1, 2, 1);
        Tensor upstream = q.detach().addScalar(0.2f);
        assertTensorGradients(q, x -> AttentionOps.causal(x, k, v, 2, 1, 0.5f), upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(k, x -> AttentionOps.causal(q, x, v, 2, 1, 0.5f), upstream, 1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(v, x -> AttentionOps.causal(q, k, x, 2, 1, 0.5f), upstream, 1e-3f, 1e-4f, 1e-3f);
    }

    @Test
    void rotaryRepeatsTimePositionsForEachBatchAndHeadAndPreservesInverseAdjoint() {
        float[] values = new float[48];
        for (int index = 0; index < values.length; index++) { values[index] = index * 0.01f; }
        Tensor input = Tensor.from(values, 2, 2, 3, 4);
        RotaryEmbedding rotary = new RotaryEmbedding(4, 8);
        Tensor actual = rotary.apply(input, 3, 2, 1);
        Tensor packed = input.reshape(12, 4);
        assertArrayEquals(rotary.apply(packed, 3, 4, 1).toFloatArray(), actual.toFloatArray());
        assertTensorGradients(input, x -> rotary.apply(x, 3, 2), input.detach(), 1e-3f, 1e-4f, 1e-3f);
        assertThrows(IllegalArgumentException.class, () -> rotary.apply(input, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> rotary.apply(input, 3, 1));
    }

    @Test
    void layoutValidationRejectsMixedRanksAxesAndNonmatchingSequenceBoundaries() {
        Tensor heads = new Tensor(2, 2, 3, 4), kv = new Tensor(2, 1, 3, 4);
        assertThrows(IllegalArgumentException.class, () -> HeadLayout.split(new Tensor(2, 3), 1));
        assertThrows(IllegalArgumentException.class, () -> HeadLayout.split(new Tensor(2, 3, 4), 3));
        assertThrows(IllegalArgumentException.class, () -> HeadLayout.merge(heads, 1));
        assertThrows(IllegalArgumentException.class, () -> AttentionOps.causal(heads, kv, kv, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> AttentionOps.causal(heads, kv, kv, 2, 1, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> AttentionOps.causal(heads, kv.reshape(6, 4), kv, 2, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> AttentionOps.causal(heads, new Tensor(1, 1, 3, 4), new Tensor(1, 1, 3, 4), 2, 1, 1));
    }
}
