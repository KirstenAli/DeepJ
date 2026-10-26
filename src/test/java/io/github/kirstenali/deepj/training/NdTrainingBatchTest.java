package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.List;
import java.util.stream.Stream;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class NdTrainingBatchTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }
    static Stream<String> names() { return families(); }

    @ParameterizedTest
    @MethodSource("names")
    void oneDenseGraphMatchesPerSequenceMaskedTrainingLossesGradientsAndUpdates(String family) {
        Observed dense = new Observed(model(family, 91), true);
        Observed sequential = new Observed(model(family, 91), false);
        Batch batch = new Batch(inputs(), targets(), masks());
        Trainer actual = CausalLMTraining.trainer(dense, ignored -> batch, 0.001f);
        Trainer expected = CausalLMTraining.trainer(sequential, ignored -> batch, 0.001f);
        for (int step = 0; step < 3; step++) {
            assertEquals(expected.trainStep(2), actual.trainStep(2), 1e-6f);
            compareParameters(sequential, dense);
        }
        assertEquals(3, dense.batchCalls);
        assertEquals(0, dense.singleCalls);
        assertEquals(6, sequential.singleCalls);
        assertEquals(0, sequential.batchCalls);
    }

    private static void compareParameters(Observed expected, Observed actual) {
        for (int index = 0; index < expected.parameters().size(); index++) {
            Parameter first = expected.parameters().get(index), second = actual.parameters().get(index);
            assertArrayEquals(first.value.toFloatArray(), second.value.toFloatArray(), 1e-6f);
            assertArrayEquals(first.grad().toFloatArray(), second.grad().toFloatArray(), 1e-6f);
        }
    }

    @Test
    void invalidBatchCountsMasksAndLogitRanksHaveExplicitPreflight() {
        Batch batch = new Batch(inputs(), targets(), masks());
        assertThrows(IllegalArgumentException.class, () -> BatchedCausalLoss.requireBatch(null, 2));
        assertThrows(IllegalArgumentException.class, () -> BatchedCausalLoss.requireBatch(batch, 3));
        assertThrows(IllegalArgumentException.class,
                () -> BatchedCausalLoss.requireBatch(new Batch(inputs(), targets(), new boolean[1][]), 2));
        assertThrows(IllegalArgumentException.class,
                () -> BatchedCausalLoss.sum(new Tensor(8, 19), batch, 2));
        CausalLM fallback = new Observed(model("nova", 4), false);
        assertFalse(fallback.supportsBatchedForward());
    }

    private static final class Observed implements CausalLM {
        private final DecoderOnlyModel model;
        private final boolean dense;
        private int batchCalls, singleCalls;
        Observed(DecoderOnlyModel model, boolean dense) {
            this.model = model;
            this.dense = dense;
        }
        @Override public Tensor forward(int[] ids) {
            singleCalls++;
            return model.forward(ids);
        }
        @Override public Tensor forwardBatch(int[][] ids) {
            batchCalls++;
            return model.forwardBatch(ids);
        }
        @Override public boolean supportsBatchedForward() { return dense; }
        @Override public float gradClipNorm() { return model.gradClipNorm(); }
        @Override public List<Parameter> parameters() { return model.parameters(); }
    }
}
