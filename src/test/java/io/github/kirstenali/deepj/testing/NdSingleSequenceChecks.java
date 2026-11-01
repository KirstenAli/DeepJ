package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

/** Single-sequence specialization must preserve the genuine singleton-batch ND computation. */
public final class NdSingleSequenceChecks {
    private NdSingleSequenceChecks() { }

    public static void exactSingletonBatch(String family, DType dtype) {
        var policy = new ParameterFactory(dtype);
        DecoderOnlyModel single = model(family, 174, policy), batch = model(family, 174, policy);
        Tensor first = single.forward(inputs()[0]), second = batch.forwardBatch(new int[][]{inputs()[0]});
        assertArrayEquals(new int[]{4, 19}, first.shape());
        assertArrayEquals(new int[]{1, 4, 19}, second.shape());
        assertArrayEquals(first.toFloatArray(), second.toFloatArray());
        Tensor firstLoss = CrossEntropyLoss.forward(first, targets()[0], masks()[0]);
        Tensor secondLoss = CrossEntropyLoss.forward(second.select(0, 0), targets()[0], masks()[0]);
        assertEquals(firstLoss.item(), secondLoss.item());
        firstLoss.backward();
        secondLoss.backward();
        gradients(single, batch);
    }

    private static void gradients(DecoderOnlyModel single, DecoderOnlyModel batch) {
        var expected = single.parameters();
        var actual = batch.parameters();
        assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            Tensor first = expected.get(index).grad(), second = actual.get(index).grad();
            assertEquals(DType.FLOAT32, first.dtype());
            assertEquals(DType.FLOAT32, second.dtype());
            assertArrayEquals(first.shape(), second.shape());
            assertArrayEquals(first.toFloatArray(), second.toFloatArray(), "parameter=" + index);
        }
    }
}
