package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.data.*;
import io.github.kirstenali.deepj.models.*;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.*;
import java.io.IOException;
import java.nio.file.*;
import static org.junit.jupiter.api.Assertions.*;

/** Same actual-model file/save-resume case on CPU and physical Metal,with exact state comparisons. */
public final class ModelResumeCase {
    private ModelResumeCase() {}

    public static void verify(DecoderOnlyModel original, Path directory) throws IOException {
        AdamW optimizer = AdamW.defaultAdamW(0.001f), resumed = AdamW.defaultAdamW(0.001f);
        Cursor data = new Cursor(), restoredData = new Cursor();
        float loss = advance(original, optimizer, data, 3);
        Path model = directory.resolve("model.dj"), training = directory.resolve("training.dj");
        original.save(model);
        DecoderOnlyModel restored = ModelLoader.load(model);
        TrainingProgress progress = save(original, optimizer, data, loss, training);
        assertEquals(progress, TrainingCheckpoint.load(restored, resumed, restoredData, schedule(), training));
        assertEquals(optimizer.stepCount(), resumed.stepCount());
        compareContinuation(original, restored, optimizer, resumed, data, restoredData, directory);
    }

    private static void compareContinuation(DecoderOnlyModel original, DecoderOnlyModel restored,
                                             AdamW optimizer, AdamW resumed, Cursor data, Cursor restoredData,
                                             Path directory) throws IOException {
        float expected = advance(original, optimizer, data, 2), actual = advance(restored, resumed, restoredData, 2);
        assertEquals(expected, actual, "Resumed model loss bits");
        Path first = directory.resolve("expected.dj"), second = directory.resolve("actual.dj");
        save(original, optimizer, data, expected, first);
        save(restored, resumed, restoredData, actual, second);
        assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second), "Complete optimizer checkpoint");
        for (int index = 0; index < original.parameters().size(); index++) {
            assertArrayEquals(original.parameters().get(index).grad().toFloatArray(),
                    restored.parameters().get(index).grad().toFloatArray(), "gradient=" + index);
        }
    }

    private static float advance(DecoderOnlyModel model, AdamW optimizer, Cursor data, int steps) {
        Trainer trainer = CausalLMTraining.trainer(model, data, optimizer);
        float loss = 0;
        for (int step = 0; step < steps; step++) {
            loss = trainer.trainStep(2);
            assertTrue(Float.isFinite(loss));
        }
        return loss;
    }

    private static TrainingProgress save(DecoderOnlyModel model, AdamW optimizer, Cursor data,
                                          float loss, Path path) throws IOException {
        var progress = new TrainingProgress((int) optimizer.stepCount(), loss, loss);
        TrainingCheckpoint.save(model, optimizer, data, progress, schedule(), path);
        return progress;
    }

    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(0.001f, 0.0001f, 1, 20);
    }

    private static final class Cursor implements StatefulBatchSource {
        private long position;
        @Override public Batch nextBatch(int batchSize) {
            assertEquals(2, batchSize);
            int[][] inputs = NdModelFixtures.inputs();
            inputs[0][0] = (int) (position++ % 10) + 1;
            return new Batch(inputs, NdModelFixtures.targets(), NdModelFixtures.masks());
        }
        @Override public long randomState() { return position; }
        @Override public void restoreRandomState(long state) { position = state; }
    }
}
