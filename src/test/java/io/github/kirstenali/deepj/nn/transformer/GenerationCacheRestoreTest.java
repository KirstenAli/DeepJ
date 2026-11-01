package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertThrows;

class GenerationCacheRestoreTest {
    @TempDir Path directory;

    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void loadingAnInferenceCheckpointInvalidatesAnOpenGenerationSession() throws Exception {
        DecoderOnlyModel model = model();
        Path file = directory.resolve("model.dj");
        model.save(file);
        try (var session = model.openGeneration()) {
            session.apply(new int[]{1, 2});
            model.load(file);
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1, 2, 3}));
        }
    }

    @Test
    void restoringTrainingStateAlsoInvalidatesAnOpenGenerationSession() throws Exception {
        DecoderOnlyModel model = model();
        Path file = directory.resolve("training.dj");
        var optimizer = AdamW.defaultAdamW(1e-4f);
        var schedule = new CosineLearningRateSchedule(1e-4f, 1e-5f, 0, 10);
        TrainingCheckpoint.save(model, optimizer, SingleBatch.INSTANCE,
                new TrainingProgress(0, 1, 1), schedule, file);
        try (var session = model.openGeneration()) {
            session.apply(new int[]{1, 2});
            TrainingCheckpoint.load(model, optimizer, SingleBatch.INSTANCE, schedule, file);
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1, 2, 3}));
        }
    }

    private static DecoderOnlyModel model() {
        return InferenceTestModels.create(InferenceTestModels.Kind.NOVA, 8, 16);
    }

    private enum SingleBatch implements StatefulBatchSource {
        INSTANCE;

        @Override
        public Batch nextBatch(int count) { return new Batch(new int[][]{{1, 2}}, new int[][]{{2, 3}}); }

        @Override
        public long randomState() { return 0; }

        @Override
        public void restoreRandomState(long state) {}
    }
}
