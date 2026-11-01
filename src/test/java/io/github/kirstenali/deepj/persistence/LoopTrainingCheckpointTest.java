package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class LoopTrainingCheckpointTest {

    @TempDir Path directory;
    private Path corpus, saved;
    private final CosineLearningRateSchedule schedule = new CosineLearningRateSchedule(0.001f, 0.0001f, 1, 10);

    @BeforeEach
    void prepare() throws Exception {
        Tensor.setBackend(new CpuBackend());
        corpus = directory.resolve("corpus.txt");
        saved = directory.resolve("training.dj");
        Files.writeString(corpus, "the little dog played outside. ".repeat(20));
    }

    @Test
    void architectureAwareResumeReproducesTheNextUpdate() throws Exception {
        var original = model(2, 37L);
        var continued = model(2, 99L);
        var first = AdamW.defaultAdamW(0.001f);
        var second = AdamW.defaultAdamW(0.001f);
        try (var a = dataset(41L); var b = dataset(59L)) {
            float loss = CausalLMTraining.trainer(original, a, first).trainStep(1);
            var progress = new TrainingProgress(1, loss, loss);
            TrainingCheckpoint.save(original, first, a, progress, schedule, saved);
            assertEquals(progress, TrainingCheckpoint.load(continued, second, b, schedule, saved));
            assertEquals(CausalLMTraining.trainer(original, a, first).trainStep(1),
                    CausalLMTraining.trainer(continued, b, second).trainStep(1), 0.0f);
            assertWeightsEqual(original, continued);
        }
    }

    @Test
    void differentLoopCountIsRejectedBeforeWeightsOrStateChange() throws Exception {
        var original = model(2, 37L);
        var incompatible = model(3, 99L);
        float[] untouched = incompatible.parameters().get(0).value.rawData().clone();
        var optimizer = AdamW.defaultAdamW(0.001f);
        try (var data = dataset(41L)) {
            TrainingCheckpoint.save(original, optimizer, data, TrainingProgress.initial(), schedule, saved);
            long randomState = data.randomState();
            var error = assertThrows(IOException.class, () ->
                    TrainingCheckpoint.load(incompatible, optimizer, data, schedule, saved));
            assertTrue(error.getMessage().contains("configuration"));
            assertArrayEquals(untouched, incompatible.parameters().get(0).value.rawData());
            assertEquals(0, optimizer.stepCount());
            assertEquals(randomState, data.randomState());
        }
    }

    @Test
    void legacyParameterOnlyCheckpointsCannotResumeALoopedModel() throws Exception {
        var decoder = model(2, 37L);
        var optimizer = AdamW.defaultAdamW(0.001f);
        try (var data = dataset(41L)) {
            TrainingCheckpoint.save(decoder.parameters(), optimizer, data,
                    TrainingProgress.initial(), schedule, saved);
            assertThrows(IOException.class, () -> TrainingCheckpoint.load(decoder, optimizer, data, schedule, saved));
        }
    }

    @Test
    void architectureAwareCheckpointsRequireArchitectureAwareLoading() throws Exception {
        var decoder = model(2, 37L);
        var optimizer = AdamW.defaultAdamW(0.001f);
        try (var data = dataset(41L)) {
            TrainingCheckpoint.save(decoder, optimizer, data, TrainingProgress.initial(), schedule, saved);
            assertThrows(IOException.class, () ->
                    TrainingCheckpoint.load(decoder.parameters(), optimizer, data, schedule, saved));
        }
    }

    @Test
    void legacyNovaTrainingCheckpointsStillLoad() throws Exception {
        var nova = new io.github.kirstenali.deepj.nn.transformer.nova.DeepJNova(model(2, 37L).config().backbone(), 79L);
        var optimizer = AdamW.defaultAdamW(0.001f);
        try (var data = dataset(41L)) {
            TrainingCheckpoint.save(nova.parameters(), optimizer, data,
                    TrainingProgress.initial(), schedule, saved);
            assertEquals(TrainingProgress.initial(), TrainingCheckpoint.load(nova, optimizer, data, schedule, saved));
        }
    }

    private RandomAccessTextDataset dataset(long seed) throws IOException {
        return new RandomAccessTextDataset(corpus, new ByteTokenizer(), 4, seed);
    }

    private static DeepJLoop model(int passes, long seed) {
        return new DeepJLoop(new DeepJLoopConfig(256, 4, 8, 4, 2,
                1, 1, 1, passes, 12, true, 0.2f, 1.0f, 0.5f), seed);
    }

    private static void assertWeightsEqual(DeepJLoop a, DeepJLoop b) {
        for (int index = 0; index < a.parameters().size(); index++) {
            assertArrayEquals(a.parameters().get(index).value.rawData(),
                    b.parameters().get(index).value.rawData(), 0.0f);
        }
    }
}
