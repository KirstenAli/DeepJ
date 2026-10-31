package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.io.IOException;
import java.nio.file.*;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdLossCheckpointTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @ParameterizedTest
    @EnumSource(DType.class)
    void gpuNdLossGradientsTypedWeightsAndOptimizerFilesResumeExactly(DType dtype) throws IOException {
        Parameter original = parameter(dtype), restored = parameter(dtype);
        AdamW optimizer = AdamW.defaultAdamW(0.01f), continuation = AdamW.defaultAdamW(0.01f);
        Cursor data = new Cursor(), resumed = new Cursor();
        float loss = advance(original, optimizer, data, 7);
        Path checkpoint = save("before.dj", original, optimizer, data, loss);
        var progress = TrainingCheckpoint.load(List.of(restored), continuation, resumed, schedule(), checkpoint);
        assertEquals(new TrainingProgress(7, loss, loss), progress);
        assertContinuation(original, restored, optimizer, continuation, data, resumed);
    }

    private void assertContinuation(Parameter original, Parameter restored, AdamW optimizer, AdamW continuation,
                                      Cursor data, Cursor resumed) throws IOException {
        float expected = advance(original, optimizer, data, 4), actual = advance(restored, continuation, resumed, 4);
        assertEquals(expected, actual);
        assertEquals(11, continuation.stepCount());
        assertEquals(data.randomState(), resumed.randomState());
        assertArrayEquals(original.value.toFloatArray(), restored.value.toFloatArray());
        assertArrayEquals(original.grad().toFloatArray(), restored.grad().toFloatArray());
        assertArrayEquals(Files.readAllBytes(save("expected.dj", original, optimizer, data, expected)),
                Files.readAllBytes(save("actual.dj", restored, continuation, resumed, actual)));
    }

    private Path save(String name, Parameter parameter, AdamW optimizer, Cursor data, float loss) throws IOException {
        Path file = directory.resolve(name);
        TrainingCheckpoint.save(List.of(parameter), optimizer, data,
                new TrainingProgress((int) optimizer.stepCount(), loss, loss), schedule(), file);
        return file;
    }

    private static float advance(Parameter parameter, AdamW optimizer, Cursor data, int steps) {
        float loss = 0;
        for (int step = 0; step < steps; step++) {
            parameter.zeroGrad();
            int target = (int) (data.randomState() % 3);
            data.nextBatch(1);
            Tensor objective = CrossEntropyLoss.forward(parameter.value, new int[]{target, (target + 1) % 3},
                    new boolean[]{true, true});
            loss = objective.item();
            assertTrue(Float.isFinite(loss));
            objective.backward();
            optimizer.step(List.of(parameter));
        }
        return loss;
    }

    private static Parameter parameter(DType dtype) {
        return new Parameter(Tensor.from(new float[]{0.4f, -0.1f, 0.2f, 0.1f, -0.5f, 0.3f}, dtype, 1, 2, 3));
    }
    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(0.01f, 0.001f, 1, 20);
    }
    private static final class Cursor implements StatefulBatchSource {
        private long position;
        @Override public Batch nextBatch(int size) {
            position++;
            return new Batch(new int[][]{{0}}, new int[][]{{1}});
        }
        @Override public long randomState() { return position; }
        @Override public void restoreRandomState(long state) { position = state; }
    }
}
