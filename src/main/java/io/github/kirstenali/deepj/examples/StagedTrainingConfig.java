package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.TransformerConfig;

import java.nio.file.Path;

interface StagedTrainingConfig {
    DeepJ100MStage stage();
    TransformerConfig architecture();
    TrainingOptions training();
    Path corpus();
    Path base();
    Path output();
    Path initialModel();
    Path resume();
    long validationBytes();
    long seed();
    String modelName();
    DecoderOnlyModel createModel();
}
