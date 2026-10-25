package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class MetalOptimizerCheckpointTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @Test
    void savingAndRestoringLazyMomentsPreservesTheNextUpdateExactly() throws Exception {
        Path corpus = Files.writeString(directory.resolve("corpus.txt"), "abcdefghij".repeat(50));
        var dataset = new RandomAccessTextDataset(corpus, new ByteTokenizer(), 8, 71);
        Parameter original = parameter();
        AdamW optimizer = AdamW.defaultAdamW(1e-3f);
        optimizer.step(List.of(original));
        Path checkpoint = directory.resolve("training.dj");
        save(checkpoint, original, optimizer, dataset);
        Parameter restored = parameter();
        AdamW resumed = AdamW.defaultAdamW(1e-3f);
        TrainingCheckpoint.load(List.of(restored), resumed, dataset, schedule(), checkpoint);
        assertLazy(resumed.state(List.of(restored)));
        assertNextUpdateMatches(original, optimizer, restored, resumed);
    }

    private static void assertNextUpdateMatches(Parameter original, AdamW optimizer,
                                               Parameter restored, AdamW resumed) {
        optimizer.step(List.of(original));
        resumed.step(List.of(restored));
        assertLazy(optimizer.state(List.of(original)));
        assertLazy(resumed.state(List.of(restored)));
        assertArrayEquals(original.value.data(), restored.value.data());
        assertSameMoments(optimizer.state(List.of(original)), resumed.state(List.of(restored)));
    }

    @Test
    void cpuBackendCanConsumeDeviceMomentsWithoutLosingTheirValues() {
        Parameter parameter = parameter();
        AdamW optimizer = AdamW.defaultAdamW(1e-3f);
        optimizer.step(List.of(parameter));
        Tensor.setBackend(cpu);
        optimizer.step(List.of(parameter));
        assertEquals(2, optimizer.stepCount());
        assertArrayEquals(new float[]{0.19f, -0.19f},
                optimizer.state(List.of(parameter)).firstMoments().get(0).data(), 1e-7f);
        Tensor.setBackend(metal);
        optimizer.step(List.of(parameter));
        assertEquals(3, optimizer.stepCount());
        assertArrayEquals(new float[]{0.271f, -0.271f},
                optimizer.state(List.of(parameter)).firstMoments().get(0).data(), 1e-7f);
    }

    private static void save(Path path, Parameter parameter, AdamW optimizer,
                             RandomAccessTextDataset dataset) throws Exception {
        var parameters = List.of(parameter);
        assertLazy(optimizer.state(parameters));
        TrainingCheckpoint.save(parameters, optimizer, dataset,
                new TrainingProgress(1, 2, 2), schedule(), path);
        assertNotNull(optimizer.state(parameters).firstMoments().get(0).data);
    }

    private static void assertLazy(AdamW.State state) {
        assertNull(state.firstMoments().get(0).data);
        assertNull(state.secondMoments().get(0).data);
    }

    private static void assertSameMoments(AdamW.State expected, AdamW.State actual) {
        assertEquals(expected.step(), actual.step());
        assertArrayEquals(expected.firstMoments().get(0).data(), actual.firstMoments().get(0).data());
        assertArrayEquals(expected.secondMoments().get(0).data(), actual.secondMoments().get(0).data());
    }

    private static Parameter parameter() {
        Parameter parameter = new Parameter(Tensor.ones(1, 2));
        parameter.setGradient(Tensor.from2D(new float[][]{{1, -1}}));
        return parameter;
    }

    private static CosineLearningRateSchedule schedule() {
        return new CosineLearningRateSchedule(1e-3f, 1e-4f, 2, 10);
    }
}
