package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.loop.DeepJLoop;
import io.github.kirstenali.deepj.models.loop.DeepJLoopConfig;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.*;

record DeepJLoop100MConfig(DeepJ100MStage stage, DeepJLoopConfig architecture, TrainingOptions training,
                           Path corpus, Path base, Path output, Path initialModel, Path resume,
                           long validationBytes, long seed) implements StagedTrainingConfig {

    static DeepJLoop100MConfig fromSystemProperties(DeepJ100MStage stage) {
        Path base = path("deepj.base", defaultBase(stage));
        return new DeepJLoop100MConfig(stage, architectureFromProperties(),
                DeepJ100MConfig.trainingFromProperties(stage),
                path("deepj.corpus", "sample_data/deepj-90m/" + stage.corpus), base,
                path("deepj.output", "checkpoints/deepj-loop-100m/" + stage.directory),
                path("deepj.initialModel", base.resolve("model-final.dj").toString()),
                optionalPath("deepj.resume"), DeepJ100MConfig.validationBytesFromProperties(),
                longValue("deepj.seed", stage.seed));
    }

    private static String defaultBase(DeepJ100MStage stage) {
        return switch (stage) {
            case PRETRAIN -> "checkpoints/deepj-90m/pretrain";
            case MIDTRAIN -> "checkpoints/deepj-loop-100m/pretrain";
            case FINE_TUNE -> "checkpoints/deepj-loop-100m/midtrain";
        };
    }

    private static DeepJLoopConfig architectureFromProperties() {
        int loops = integer("deepj.loops", 2), core = integer("deepj.coreLayers", 12);
        return new DeepJLoopConfig(integer("deepj.vocabSize", 16_384), integer("deepj.seqLen", 1_024),
                integer("deepj.dModel", 576), integer("deepj.heads", 9), integer("deepj.kvHeads", 3),
                integer("deepj.preludeLayers", 6), core, integer("deepj.codaLayers", 6), loops,
                integer("deepj.dFF", 1_664), Boolean.parseBoolean(System.getProperty("deepj.qkNorm", "true")),
                decimal("deepj.initScale", 0.2f), decimal("deepj.gradClipNorm", 1.0f),
                decimal("deepj.coreResidualScale", (float) (1.0 / (loops * Math.sqrt(core)))));
    }

    @Override
    public String modelName() {
        return "DeepJ Loop";
    }

    @Override
    public DeepJLoop createModel() {
        return new DeepJLoop(architecture, seed);
    }
}
