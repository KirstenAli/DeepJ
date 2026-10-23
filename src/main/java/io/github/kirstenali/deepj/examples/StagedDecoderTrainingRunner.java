package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.data.SequentialTextDataset;
import io.github.kirstenali.deepj.data.StatefulTrainingDataset;
import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class StagedDecoderTrainingRunner {

    private StagedDecoderTrainingRunner() {}

    static TrainingResult run(StagedTrainingConfig config) throws Exception {
        validateFiles(config);
        if (config.stage() == DeepJ100MStage.FINE_TUNE) return fineTune(config);
        DecoderOnlyModel model = initialModel(config);
        prepareTokenizer(config, model.config());
        DecoderTrainingSession.ensureCheckpointSpace(config.output(), model);
        model.setMetadata(metadata(config));
        printModel(config, model);
        try (var dataset = dataset(config, model)) {
            return DecoderTrainingSession.run(model, dataset, config.training(), config.output(), config.resume());
        }
    }

    private static DecoderOnlyModel initialModel(StagedTrainingConfig config) throws IOException {
        if (config.stage() == DeepJ100MStage.PRETRAIN) return config.createModel();
        DecoderOnlyModel model = ModelLoader.load(config.initialModel());
        validateArchitecture(config, model.config());
        return model;
    }

    private static void validateArchitecture(StagedTrainingConfig config, TransformerConfig actual)
            throws IOException {
        if (!actual.getClass().equals(config.architecture().getClass())) {
            throw new IOException("Expected a " + config.modelName() + " checkpoint: " + config.initialModel());
        }
    }

    private static void validateFiles(StagedTrainingConfig config) throws IOException {
        if (!Files.isRegularFile(config.corpus())) throw new IOException("Missing corpus: " + config.corpus());
        if (!Files.isRegularFile(config.base().resolve("tokenizer.bpe"))) {
            throw new IOException("Missing tokenizer.bpe in " + config.base());
        }
        Files.createDirectories(config.output());
        if (config.resume() == null && Files.exists(config.output().resolve("model-final.dj"))) {
            throw new IOException("Output already contains a model; set deepj.resume or choose another output");
        }
        if (config.resume() == null && Files.exists(config.output().resolve("training-latest.dj"))) {
            throw new IOException("Output has resumable training; set deepj.resume or choose another output");
        }
    }

    private static void prepareTokenizer(StagedTrainingConfig config, TransformerConfig architecture)
            throws IOException {
        Path source = config.base().resolve("tokenizer.bpe");
        Path target = config.output().resolve("tokenizer.bpe");
        var tokenizer = new BPETokenizer(BPEModelIO.load(source));
        if (tokenizer.vocabSize() != architecture.vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match model configuration");
        }
        if (Files.exists(target) && Files.mismatch(source, target) != -1L) {
            throw new IOException("Output tokenizer differs from the source tokenizer");
        }
        if (!Files.exists(target)) Files.copy(source, target);
    }

    private static StatefulTrainingDataset dataset(StagedTrainingConfig config, DecoderOnlyModel model)
            throws IOException {
        var tokenizer = new BPETokenizer(BPEModelIO.load(config.output().resolve("tokenizer.bpe")));
        int sequence = model.config().maxSeqLen();
        if (config.stage() == DeepJ100MStage.MIDTRAIN) {
            return new RandomAccessTextDataset(config.corpus(), tokenizer, sequence, config.seed());
        }
        var split = DeepJ90MDataSplit.split(config.corpus(), config.validationBytes());
        System.out.printf("Corpus: trainingBytes=%,d validationBytes=%,d%n",
                split.training().length(), split.validation().length());
        return new SequentialTextDataset(config.corpus(), tokenizer, sequence, split.training());
    }

    private static Map<String, String> metadata(StagedTrainingConfig config) {
        var values = new HashMap<String, String>();
        TrainingMetadata.add(values, config.training());
        values.put("stage", config.stage().directory);
        values.put("corpus", config.corpus().toAbsolutePath().toString());
        values.put("seed", Long.toString(config.seed()));
        values.put("datasetOrder", config.stage() == DeepJ100MStage.PRETRAIN ? "sequential" : "random");
        values.put("validationBytes", Long.toString(config.validationBytes()));
        return values;
    }

    private static void printModel(StagedTrainingConfig config, DecoderOnlyModel model) {
        long count = model.parameters().stream().mapToLong(parameter -> parameter.value.data.length).sum();
        System.out.printf("Backend: %s%nModel: %s%nParameters: %,d%nStage: %s%n",
                Tensor.backend().getClass().getSimpleName(), config.modelName(), count, config.stage().directory);
    }

    private static TrainingResult fineTune(StagedTrainingConfig config) throws Exception {
        var settings = ModelSerializer.settings(config.initialModel());
        if (settings == null) throw new IOException("Model has no embedded configuration");
        validateArchitecture(config, settings.config());
        prepareTokenizer(config, settings.config());
        System.out.println("Model: " + config.modelName());
        return ResponseOnlyFineTuner.run(responseConfig(config));
    }

    private static ResponseFineTuningConfig responseConfig(StagedTrainingConfig config) {
        var options = config.training();
        var files = new ResponseFineTuningConfig.FilesConfig(config.base(), config.output(),
                config.initialModel(), config.resume());
        var training = new ResponseFineTuningConfig.Training(options.steps(), options.batchSize(),
                options.peakLearningRate(), options.minimumLearningRate(), options.warmupSteps(),
                options.logEvery(), options.checkpointEvery(), options.releaseEvery());
        return new ResponseFineTuningConfig(files, training,
                List.of(new ResponseOnlyTextDataset.Source(config.corpus(), 1)), config.seed());
    }
}
