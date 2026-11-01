package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

import java.nio.file.Path;

public record DeepJPrismTrainingConfig(
        FilesConfig files,
        Architecture architecture,
        Training training,
        TokenizerConfig tokenizer,
        long seed
) {

    public DeepJPrismTrainingConfig {
        if (files == null || architecture == null || training == null || tokenizer == null) {
            throw new IllegalArgumentException("configuration sections must not be null");
        }
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

        DeepJPrismConfig modelConfig(int vocabSize) {
            return new DeepJPrismConfig(vocabSize, sequenceLength, dModel, heads, layers,
                    dFF, qRank, kvRank, initScale, gradClipNorm);
        }
    }

    public record Training(int steps, int batchSize, float peakLearningRate,
                           float minimumLearningRate, int warmupSteps, int logEvery,
                           int checkpointEvery, int releaseEvery) implements TrainingMetadata.Settings {

        public Training {
            if (steps <= 0 || batchSize <= 0 || logEvery <= 0) {
                throw new IllegalArgumentException("steps, batchSize, and logEvery must be positive");
            }
            if (checkpointEvery < 0 || releaseEvery < 0) {
                throw new IllegalArgumentException("checkpointEvery and releaseEvery must be non-negative");
            }
            new CosineLearningRateSchedule(peakLearningRate, minimumLearningRate, warmupSteps, steps);
        }

    }

    public record TokenizerConfig(int vocabSize, int sampleMiB) {

        public TokenizerConfig {
            if (vocabSize <= 261) throw new IllegalArgumentException("vocabSize must be > 261");
            if (sampleMiB <= 0 || sampleMiB > 50) {
                throw new IllegalArgumentException("sampleMiB must be in [1, 50]");
            }
        }

        int sampleChars() {
            return Math.multiplyExact(sampleMiB, 1024 * 1024);
        }
    }

}
