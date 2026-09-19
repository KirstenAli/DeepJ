package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainingCheckpointTest {

    @TempDir
    Path temporaryDirectory;

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void restoresModelOptimizerProgressAndDataset() throws Exception {
        Path corpus = corpus();
        Path checkpoint = temporaryDirectory.resolve("training.dj");
        List<Parameter> source = parameters(1.0f, -2.0f);
        AdamW sourceOptimizer = trainedOptimizer(source);
        try (RandomAccessTextDataset sourceData = dataset(corpus, 42L)) {
            sourceData.nextBatch(1);
            save(checkpoint, source, sourceOptimizer, sourceData);
            Batch expectedBatch = sourceData.nextBatch(1);
            assertRestored(checkpoint, corpus, source, sourceOptimizer, expectedBatch);
        }
    }

    private void assertRestored(Path checkpoint, Path corpus, List<Parameter> source,
                                AdamW sourceOptimizer, Batch expectedBatch) throws Exception {
        List<Parameter> restored = parameters(0.0f, 0.0f);
        AdamW restoredOptimizer = AdamW.defaultAdamW(schedule().learningRate(0));
        try (RandomAccessTextDataset restoredData = dataset(corpus, 99L)) {
            TrainingProgress progress = TrainingCheckpoint.load(
                    restored, restoredOptimizer, restoredData, schedule(), checkpoint);
            assertEquals(new TrainingProgress(1, 2.5f, 2.25f), progress);
            assertBatchEquals(expectedBatch, restoredData.nextBatch(1));
            assertOptimizerContinuation(source, sourceOptimizer, restored, restoredOptimizer);
        }
    }

    private static void assertOptimizerContinuation(List<Parameter> source, AdamW sourceOptimizer,
                                                    List<Parameter> restored, AdamW restoredOptimizer) {
        setGradients(source, 0.25f);
        setGradients(restored, 0.25f);
        sourceOptimizer.step(source);
        restoredOptimizer.step(restored);
        assertEquals(sourceOptimizer.stepCount(), restoredOptimizer.stepCount());
        assertArrayEquals(source.get(0).value.data, restored.get(0).value.data, 0.0f);
    }

    private static AdamW trainedOptimizer(List<Parameter> params) {
        AdamW optimizer = AdamW.defaultAdamW(schedule().learningRate(0));
        setGradients(params, 0.5f);
        optimizer.step(params);
        optimizer.setLr(schedule().learningRate(1));
        return optimizer;
    }

    private static void save(Path path, List<Parameter> params, AdamW optimizer,
                             RandomAccessTextDataset dataset) throws Exception {
        TrainingProgress progress = new TrainingProgress(1, 2.5f, 2.25f);
        TrainingCheckpoint.save(params, optimizer, dataset, progress, schedule(), path);
        assertTrue(TrainingCheckpoint.matches(path));
    }

    private static void setGradients(List<Parameter> params, float value) {
        for (Parameter parameter : params) {
            parameter.grad = row(value, -value);
        }
    }

    private static List<Parameter> parameters(float first, float second) {
        return List.of(new Parameter(row(first, second)));
    }

    private static Tensor row(float... values) {
        return TensorAdapters.unpackF32(values, 1, values.length);
    }

    private RandomAccessTextDataset dataset(Path corpus, long seed) throws Exception {
        return new RandomAccessTextDataset(corpus, new ByteTokenizer(), 8, seed);
    }

    private Path corpus() throws Exception {
        return Files.writeString(temporaryDirectory.resolve("corpus.txt"),
                "abcdefghijklmnopqrstuvwxyz\n".repeat(100));
    }

    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(1e-3f, 1e-4f, 2, 10);
    }

    private static void assertBatchEquals(Batch expected, Batch actual) {
        assertArrayEquals(expected.x()[0], actual.x()[0]);
        assertArrayEquals(expected.y()[0], actual.y()[0]);
    }
}
