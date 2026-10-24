package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNova;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.*;

record DeepJ100MConfig(DeepJ100MStage stage, DeepJNovaConfig architecture, TrainingOptions training,
                       Path corpus, Path base, Path output, Path initialModel, Path resume,
                       long validationBytes, long seed) implements StagedTrainingConfig {

    static DeepJ100MConfig fromSystemProperties(DeepJ100MStage stage) {
        Path base = path("deepj.base", "checkpoints/" + stage.base);
        return new DeepJ100MConfig(stage, architectureFromProperties(), trainingFromProperties(stage),
                path("deepj.corpus", "sample_data/deepj-90m/" + stage.corpus), base,
                path("deepj.output", "checkpoints/deepj-100m/" + stage.directory),
                path("deepj.initialModel", base.resolve("model-final.dj").toString()),
                optionalPath("deepj.resume"), validationBytesFromProperties(),
                longValue("deepj.seed", stage.seed));
    }

    static long validationBytesFromProperties() {
        return Math.multiplyExact((long) integer("deepj.validationMiB", 64), 1_024L * 1_024L);
    }

    private static DeepJNovaConfig architectureFromProperties() {
        return new DeepJNovaConfig(integer("deepj.vocabSize", 16_384), integer("deepj.seqLen", 1_024),
                integer("deepj.dModel", 576), integer("deepj.heads", 9), integer("deepj.kvHeads", 3),
                integer("deepj.layers", 24), integer("deepj.dFF", 1_664),
                Boolean.parseBoolean(System.getProperty("deepj.qkNorm", "true")),
                decimal("deepj.initScale", 0.2f), decimal("deepj.gradClipNorm", 1.0f));
    }

    static TrainingOptions trainingFromProperties(DeepJ100MStage stage) {
        return new TrainingOptions(integer("deepj.steps", stage.steps), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", stage.learningRate),
                decimal("deepj.minLearningRate", stage.learningRate / 10.0f),
                integer("deepj.warmupSteps", stage.warmup), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 1));
    }

    @Override
    public String modelName() {
        return "DeepJ Nova";
    }

    @Override
    public DeepJNova createModel() {
        return new DeepJNova(architecture, seed);
    }
}
