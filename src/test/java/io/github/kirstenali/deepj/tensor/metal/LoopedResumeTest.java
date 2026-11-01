package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingProgress;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LoopedResumeTest extends MetalDifferentialTestSupport {

    @TempDir Path directory;

    @Test
    void metalCheckpointContinuesTheSameSharedCoreUpdates() throws Exception {
        Path corpus = directory.resolve("corpus.txt"), checkpoint = directory.resolve("resume.dj");
        Files.writeString(corpus, "the little dog played outside. ".repeat(100));
        var config = new DeepJLoopConfig(256, 256, 64, 1, 1, 0, 1, 0, 2, 96);
        var original = new DeepJLoop(config, 83L);
        var optimizer = AdamW.defaultAdamW(0.0001f);
        var schedule = new CosineLearningRateSchedule(0.0001f, 0.00001f, 1, 10);
        try (var data = dataset(corpus, 89L)) {
            Trainer trainer = CausalLMTraining.trainer(original, data, optimizer);
            float lastLoss = steps(trainer, 3);
            TrainingCheckpoint.save(original, optimizer, data,
                    new TrainingProgress(3, lastLoss, lastLoss), schedule, checkpoint);
            float nextLoss = trainer.trainStep(1);
            List<Tensor> expected = weights(original);
            assertContinuation(config, corpus, checkpoint, schedule, nextLoss, expected);
        }
    }

    private void assertContinuation(DeepJLoopConfig config, Path corpus, Path checkpoint,
                                    CosineLearningRateSchedule schedule, float loss, List<Tensor> expected)
            throws Exception {
        var restored = new DeepJLoop(config, 97L);
        var optimizer = AdamW.defaultAdamW(0.0001f);
        try (var data = dataset(corpus, 101L)) {
            TrainingCheckpoint.load(restored, optimizer, data, schedule, checkpoint);
            assertEquals(loss, CausalLMTraining.trainer(restored, data, optimizer).trainStep(1), 0.0f);
            assertGradientListsClose(expected, weights(restored), 0.0f, 0.0f);
        }
    }

    private static RandomAccessTextDataset dataset(Path corpus, long seed) throws Exception {
        return new RandomAccessTextDataset(corpus, new ByteTokenizer(), 256, seed);
    }

    private static float steps(Trainer trainer, int count) {
        float loss = Float.NaN;
        for (int index = 0; index < count; index++) {
            loss = trainer.trainStep(1);
            Tensor.backend().releaseTemporaryResources();
        }
        return loss;
    }

    private static List<Tensor> weights(DeepJLoop model) {
        return model.parameters().stream().map(p -> new Tensor(p.value)).toList();
    }
}
