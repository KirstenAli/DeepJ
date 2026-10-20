package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.data.SequentialTextDataset;
import io.github.kirstenali.deepj.data.StatefulTrainingDataset;
import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNovaParameterCount;
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

final class DeepJNovaTrainingRunner {

    private DeepJNovaTrainingRunner() {}

    static TrainingResult run(DeepJ100MConfig config) throws Exception {
        validateFiles(config);
        if (config.stage() == DeepJ100MStage.FINE_TUNE) return fineTune(config);
        DeepJNova model = initialModel(config);
        prepareTokenizer(config, model.config());
        DecoderTrainingSession.ensureCheckpointSpace(config.output(), model);
        model.setMetadata(metadata(config));
        printModel(config, model.config());
        try (var dataset = dataset(config, model)) {
            return DecoderTrainingSession.run(model, dataset, config.training(), config.output(), config.resume());
        }
    }

    private static DeepJNova initialModel(DeepJ100MConfig config) throws IOException {
        if (config.stage() == DeepJ100MStage.PRETRAIN) return new DeepJNova(config.architecture(), config.seed());
        return loadNova(config.initialModel());
    }

    private static DeepJNova loadNova(Path path) throws IOException {
        if (ModelLoader.load(path) instanceof DeepJNova nova) return nova;
        throw new IOException("Expected a DeepJ Nova checkpoint: " + path);
    }

    private static void validateFiles(DeepJ100MConfig config) throws IOException {
        if (!Files.isRegularFile(config.corpus())) throw new IOException("Missing corpus: " + config.corpus());
        if (!Files.isRegularFile(config.base().resolve("tokenizer.bpe"))) {
            throw new IOException("Missing Frisky/Nova tokenizer.bpe in " + config.base());
        }
        Files.createDirectories(config.output());
        if (config.resume() == null && Files.exists(config.output().resolve("model-final.dj"))) {
            throw new IOException("Output already contains a model; set deepj.resume or choose another output");
        }
        if (config.resume() == null && Files.exists(config.output().resolve("training-latest.dj"))) {
            throw new IOException("Output has resumable training; set deepj.resume or choose another output");
        }
    }

    private static void prepareTokenizer(DeepJ100MConfig config, DeepJNovaConfig architecture) throws IOException {
        Path source = config.base().resolve("tokenizer.bpe");
        Path target = config.output().resolve("tokenizer.bpe");
        var tokenizer = new BPETokenizer(BPEModelIO.load(source));
        if (tokenizer.vocabSize() != architecture.vocabSize()) {
            throw new IOException("Tokenizer vocabulary does not match Nova configuration");
        }
        if (Files.exists(target) && Files.mismatch(source, target) != -1L) {
            throw new IOException("Output tokenizer differs from the source tokenizer");
        }
        if (!Files.exists(target)) Files.copy(source, target);
    }

    private static StatefulTrainingDataset dataset(DeepJ100MConfig config, DeepJNova model) throws IOException {
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

    private static Map<String, String> metadata(DeepJ100MConfig config) {
        var values = new HashMap<String, String>();
        TrainingMetadata.add(values, config.training());
        values.put("stage", config.stage().directory);
        values.put("corpus", config.corpus().toAbsolutePath().toString());
        values.put("seed", Long.toString(config.seed()));
        values.put("datasetOrder", config.stage() == DeepJ100MStage.PRETRAIN ? "sequential" : "random");
        values.put("validationBytes", Long.toString(config.validationBytes()));
        return values;
    }

    private static void printModel(DeepJ100MConfig config, DeepJNovaConfig architecture) {
        System.out.printf("Backend: %s%nModel: DeepJ Nova%nParameters: %,d%nStage: %s%n",
                Tensor.backend().getClass().getSimpleName(), DeepJNovaParameterCount.count(architecture),
                config.stage().directory);
    }

    private static TrainingResult fineTune(DeepJ100MConfig config) throws Exception {
        var settings = ModelSerializer.settings(config.initialModel());
        if (settings == null || !(settings.config() instanceof DeepJNovaConfig architecture)) {
            throw new IOException("Expected a DeepJ Nova checkpoint: " + config.initialModel());
        }
        prepareTokenizer(config, architecture);
        printModel(config, architecture);
        return ResponseOnlyFineTuner.run(responseConfig(config));
    }

    private static ResponseFineTuningConfig responseConfig(DeepJ100MConfig config) {
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
