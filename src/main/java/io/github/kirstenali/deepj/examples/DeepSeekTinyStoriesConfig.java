package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

import java.nio.file.Path;

public record DeepSeekTinyStoriesConfig(
        FilesConfig files,
        Architecture architecture,
        Training training,
        TokenizerConfig tokenizer,
        long seed
) {

    public DeepSeekTinyStoriesConfig {
        if (files == null || architecture == null || training == null || tokenizer == null) {
            throw new IllegalArgumentException("configuration sections must not be null");
        }
    }

    public static DeepSeekTinyStoriesConfig fromSystemProperties() {
        return new DeepSeekTinyStoriesConfig(
                FilesConfig.fromSystemProperties(),
                Architecture.fromSystemProperties(),
                Training.fromSystemProperties(),
                TokenizerConfig.fromSystemProperties(),
                longProperty("deepj.seed", 42L));
    }

    public DeepSeekConfig modelConfig(int vocabSize) {
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
                    pathProperty("deepj.corpus", "sample_data/TinyStories-train.txt"),
                    pathProperty("deepj.output", "checkpoints/tinystories-deepseek"),
                    optionalPathProperty("deepj.resume"));
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
            return new Architecture(intProperty("deepj.seqLen", 128), intProperty("deepj.dModel", 128),
                    intProperty("deepj.heads", 4), intProperty("deepj.layers", 4),
                    intProperty("deepj.dFF", 384), intProperty("deepj.qRank", 64),
                    intProperty("deepj.kvRank", 32), floatProperty("deepj.initScale", 0.2f),
                    floatProperty("deepj.gradClipNorm", 1.0f));
        }

        DeepSeekConfig modelConfig(int vocabSize) {
            return new DeepSeekConfig(vocabSize, sequenceLength, dModel, heads, layers,
                    dFF, qRank, kvRank, initScale, gradClipNorm);
        }
    }

    public record Training(int steps, int batchSize, float peakLearningRate,
                           float minimumLearningRate, int warmupSteps, int logEvery,
                           int checkpointEvery, int releaseEvery) {

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
            return new Training(intProperty("deepj.steps", 10_000), intProperty("deepj.batchSize", 1),
                    floatProperty("deepj.learningRate", 3e-4f),
                    floatProperty("deepj.minLearningRate", 3e-5f),
                    intProperty("deepj.warmupSteps", 200), intProperty("deepj.logEvery", 10),
                    intProperty("deepj.checkpointEvery", 500), intProperty("deepj.releaseEvery", 25));
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
            return new TokenizerConfig(intProperty("deepj.vocabSize", 2_048),
                    intProperty("deepj.tokenizerSampleMiB", 16));
        }

        int sampleChars() {
            return Math.multiplyExact(sampleMiB, 1024 * 1024);
        }
    }

    private static int intProperty(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }

    private static long longProperty(String name, long fallback) {
        return Long.parseLong(System.getProperty(name, Long.toString(fallback)));
    }

    private static float floatProperty(String name, float fallback) {
        return Float.parseFloat(System.getProperty(name, Float.toString(fallback)));
    }

    private static Path pathProperty(String name, String fallback) {
        return Path.of(System.getProperty(name, fallback));
    }

    private static Path optionalPathProperty(String name) {
        String value = System.getProperty(name);
        return value == null || value.isBlank() ? null : Path.of(value);
    }
}
