package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekParameterCount;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TrainDeepJ90MTest {

    @Test
    void defaultArchitectureHasExpectedParameterCount() {
        var config = TrainDeepJ90M.configuration();
        long count = DeepSeekParameterCount.count(config.modelConfig(TrainDeepJ90M.VOCAB_SIZE));

        assertEquals(90_128_896L, count);
        assertEquals(1_024, config.architecture().sequenceLength());
        assertEquals(16_384, config.tokenizer().vocabSize());
    }

    @Test
    void pretrainingCoversChinchillaTokenBudget() {
        var config = TrainDeepJ90M.configuration();
        long tokens = (long) config.training().steps() * config.training().effectiveBatchSize()
                * config.architecture().sequenceLength();

        assertEquals(1_800_000_512L, tokens);
    }

    @Test
    void postTrainingStagesUseSeparateCorpora() {
        var mid = MidTrainDeepJ90M.configuration();
        var fine = FineTuneDeepJ90M.configuration();

        assertEquals(Path.of("sample_data/deepj-90m/midtrain-train.txt"), mid.files().corpus());
        assertEquals(Path.of("sample_data/deepj-90m/sft-train.txt"), fine.sources().get(0).path());
    }

    @Test
    void midTrainingMatchesArticleTokenExposure() {
        var config = MidTrainDeepJ90M.configuration();
        long tokens = (long) config.training().steps() * config.architecture().sequenceLength();

        assertEquals(262_144_000L, tokens);
    }

    @Test
    void fineTuningUsesOneCuratedPass() {
        assertEquals(22_500, FineTuneDeepJ90M.configuration().training().steps());
    }

    @Test
    void accumulationCanBeConfiguredWithoutChangingBatchSize() {
        withProperty("deepj.gradientAccumulationSteps", "8",
                TrainDeepJ90MTest::assertAccumulationConfiguration);
    }

    private static void assertAccumulationConfiguration() {
        var training = TrainDeepJ90M.configuration().training();
        assertEquals(1, training.batchSize());
        assertEquals(8, training.gradientAccumulationSteps());
        assertEquals(8, training.effectiveBatchSize());
    }

    private static void withProperty(String name, String value, Runnable assertion) {
        String previous = System.getProperty(name);
        System.setProperty(name, value);
        try {
            assertion.run();
        } finally {
            restoreProperty(name, previous);
        }
    }

    private static void restoreProperty(String name, String value) {
        if (value == null) System.clearProperty(name);
        else System.setProperty(name, value);
    }
}
