package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.longValue;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.optionalPath;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public record DeepJPrismTinyStoriesConfig(
        FilesConfig files,
        Architecture architecture,
        Training training,
        TokenizerConfig tokenizer,
        long seed
) {

    public DeepJPrismTinyStoriesConfig {
        if (files == null || architecture == null || training == null || tokenizer == null) {
            throw new IllegalArgumentException("configuration sections must not be null");
        }
    }

    public static DeepJPrismTinyStoriesConfig fromSystemProperties() {
        return new DeepJPrismTinyStoriesConfig(
                FilesConfig.fromSystemProperties(),
                Architecture.fromSystemProperties(),
                Training.fromSystemProperties(),
                TokenizerConfig.fromSystemProperties(),
                longValue("deepj.seed", 42L));
    }

    public DeepJPrismConfig modelConfig(int vocabSize) {
        return architecture.modelConfig(vocabSize);
    }

    public record FilesConfig(Path corpus, Path outputDirectory, Path resumeCheckpoint) {

        public FilesConfig {
            if (corpus == null || outputDirectory == null) {
                throw new IllegalArgumentException("corpus and outputDirectory must not be null");
            }
        }

        static FilesConfig fromSystemProperties() {
            return new FilesConfig(
                    path("deepj.corpus", "sample_data/TinyStories-train.txt"),
                    path("deepj.output", "checkpoints/tinystories-prism"),
                    optionalPath("deepj.resume"));
        }
    }

    public record Architecture(int sequenceLength, int dModel, int heads, int layers,
                               int dFF, int qRank, int kvRank, float initScale,
                               float gradClipNorm) {

        public Architecture {
            if (sequenceLength < 2 || dModel <= 0 || heads <= 0 || layers <= 0 || dFF <= 0) {
                throw new IllegalArgumentException("architecture dimensions must be positive");
            }
            if (qRank <= 0 || kvRank <= 0) {
                throw new IllegalArgumentException("latent ranks must be positive");
            }
        }

        static Architecture fromSystemProperties() {
            return new Architecture(integer("deepj.seqLen", 128), integer("deepj.dModel", 128),
                    integer("deepj.heads", 4), integer("deepj.layers", 4),
                    integer("deepj.dFF", 384), integer("deepj.qRank", 64),
                    integer("deepj.kvRank", 32), decimal("deepj.initScale", 0.2f),
                    decimal("deepj.gradClipNorm", 1.0f));
        }

        DeepJPrismConfig modelConfig(int vocabSize) {
            return new DeepJPrismConfig(vocabSize, sequenceLength, dModel, heads, layers,
                    dFF, qRank, kvRank, initScale, gradClipNorm);
        }
    }

    public record Training(int steps, int batchSize, float peakLearningRate,
                           float minimumLearningRate, int warmupSteps, int logEvery,
                           int checkpointEvery, int releaseEvery) implements TrainingProperties.Settings {

        public Training {
            if (steps <= 0 || batchSize <= 0 || logEvery <= 0) {
                throw new IllegalArgumentException("steps, batchSize, and logEvery must be positive");
            }
            if (checkpointEvery < 0 || releaseEvery < 0) {
                throw new IllegalArgumentException("checkpointEvery and releaseEvery must be non-negative");
            }
            new CosineLearningRateSchedule(peakLearningRate, minimumLearningRate, warmupSteps, steps);
        }

        static Training fromSystemProperties() {
            return new Training(integer("deepj.steps", 10_000), integer("deepj.batchSize", 1),
                    decimal("deepj.learningRate", 3e-4f), decimal("deepj.minLearningRate", 3e-5f),
                    integer("deepj.warmupSteps", 200), integer("deepj.logEvery", 10),
                    integer("deepj.checkpointEvery", 500), integer("deepj.releaseEvery", 25));
        }
    }

    public record TokenizerConfig(int vocabSize, int sampleMiB) {

        public TokenizerConfig {
            if (vocabSize <= 261) throw new IllegalArgumentException("vocabSize must be > 261");
            if (sampleMiB <= 0 || sampleMiB > 50) {
                throw new IllegalArgumentException("sampleMiB must be in [1, 50]");
            }
        }

        static TokenizerConfig fromSystemProperties() {
            return new TokenizerConfig(integer("deepj.vocabSize", 2_048),
                    integer("deepj.tokenizerSampleMiB", 16));
        }

        int sampleChars() {
            return Math.multiplyExact(sampleMiB, 1024 * 1024);
        }
    }

}
