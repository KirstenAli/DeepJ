package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.stream.Stream;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class NdModelBatchTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }
    static Stream<String> names() { return families(); }

    @ParameterizedTest
    @MethodSource("names")
    void batchedLogitsMatchIndependentSequencesAndRetainAllAxes(String family) {
        DecoderOnlyModel model = model(family, 91);
        try (var ignored = GradMode.noGrad()) {
            Tensor output = model.forwardBatch(inputs());
            assertArrayEquals(new int[]{2, 4, 19}, output.shape());
            assertTrue(model.supportsBatchedForward());
            for (int batch = 0; batch < 2; batch++) {
                assertArrayEquals(model.forward(inputs()[batch]).toFloatArray(),
                        output.select(0, batch).toFloatArray());
            }
            assertArrayEquals(model.forward(inputs()[0]).toFloatArray(),
                    model.forwardBatch(new int[][]{inputs()[0]}).toFloatArray());
        }
    }

    @ParameterizedTest
    @MethodSource("names")
    void gradientsPreservePerSequenceResponseMaskWeightingAndSharedParameters(String family) {
        DecoderOnlyModel actual = model(family, 92), reference = model(family, 92);
        Tensor logits = actual.forwardBatch(inputs()), loss = summedLoss(logits);
        loss.backward();
        float total = 0;
        for (int sequence = 0; sequence < 2; sequence++) {
            Tensor objective = CrossEntropyLoss.forward(reference.forward(inputs()[sequence]), targets()[sequence],
                    masks()[sequence]);
            total += objective.item();
            objective.backward();
        }
        assertEquals(total, loss.item());
        assertEquals(reference.parameters().size(), actual.parameters().size());
        compareGradients(reference, actual);
    }

    private static Tensor summedLoss(Tensor logits) {
        return CrossEntropyLoss.forward(logits.select(0, 0), targets()[0], masks()[0])
                .add(CrossEntropyLoss.forward(logits.select(0, 1), targets()[1], masks()[1]));
    }

    private static void compareGradients(DecoderOnlyModel expected, DecoderOnlyModel actual) {
        for (int index = 0; index < expected.parameters().size(); index++) {
            Parameter first = expected.parameters().get(index), second = actual.parameters().get(index);
            assertArrayEquals(first.grad().toFloatArray(), second.grad().toFloatArray(), 1e-6f, "parameter=" + index);
        }
    }

    @ParameterizedTest
    @MethodSource("names")
    void causalAndBatchBoundariesCannotLeakTokens(String family) {
        DecoderOnlyModel model = model(family, 93);
        int[][] original = inputs(), changed = inputs();
        changed[0][3] = 18;
        changed[1] = new int[]{18, 17, 16, 15};
        try (var ignored = GradMode.noGrad()) {
            Tensor before = model.forwardBatch(original), after = model.forwardBatch(changed);
            assertArrayEquals(before.select(0, 0).slice(0, 0, 3, 1).toFloatArray(),
                    after.select(0, 0).slice(0, 0, 3, 1).toFloatArray());
        }
    }

    @ParameterizedTest
    @MethodSource("names")
    void malformedDenseBatchesAndOutOfRangeIdsAreRejected(String family) {
        DecoderOnlyModel model = model(family, 94);
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(null));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[0][]));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{null}));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{{1}, {1, 2}}));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{{1}, null}));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{{}}));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{new int[9]}));
        assertThrows(IllegalArgumentException.class, () -> model.forwardBatch(new int[][]{{19}}));
        assertThrows(IllegalArgumentException.class, () -> model.forward(new int[0]));
    }
}
