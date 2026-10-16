package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.data.SequentialTextDataset;
import io.github.kirstenali.deepj.data.StatefulTrainingDataset;
import io.github.kirstenali.deepj.data.TextFileRange;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingProgress;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

final class DeepJPrismTrainingRunner {

    private static final List<String> SPECIAL_TOKENS =
            List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>");
    private static final String TOKENIZER_FILE = "tokenizer.bpe";
    private static final String LATEST_MODEL_FILE = "model-latest.dj";
    private static final String LATEST_TRAINING_FILE = "training-latest.dj";
    private static final String FINAL_MODEL_FILE = "model-final.dj";

    private DeepJPrismTrainingRunner() {}

    static TrainingResult run(DeepJPrismTrainingConfig runConfig) throws Exception {
        return run(runConfig, DatasetOrder.random());
    }

    static TrainingResult runSequential(DeepJPrismTrainingConfig runConfig)
            throws Exception {
        return run(runConfig, DatasetOrder.sequential(null));
    }

    static TrainingResult runSequential(DeepJPrismTrainingConfig runConfig,
                                        TextFileRange range) throws Exception {
        return run(runConfig, DatasetOrder.sequential(range));
    }

    private static TrainingResult run(DeepJPrismTrainingConfig runConfig,
                                      DatasetOrder order) throws Exception {
        Path output = runConfig.files().outputDirectory();
        validateCorpus(runConfig.files().corpus());
        Files.createDirectories(output);
        BPETokenizer tokenizer = prepareTokenizer(runConfig);
        DeepJPrismConfig modelConfig = runConfig.modelConfig(tokenizer.vocabSize());
        DeepJPrism model = new DeepJPrism(modelConfig, runConfig.seed());
        ensureCheckpointSpace(output, model);
        model.setMetadata(metadata(runConfig, order));
        try (StatefulTrainingDataset dataset = dataset(runConfig, tokenizer, order)) {
            return train(model, dataset, runConfig);
        }
    }

    private static StatefulTrainingDataset dataset(DeepJPrismTrainingConfig config,
                                                    BPETokenizer tokenizer,
                                                    DatasetOrder order) throws IOException {
        int length = config.architecture().sequenceLength();
        if (order.sequential()) return sequentialDataset(config, tokenizer, length, order.range());
        return new RandomAccessTextDataset(config.files().corpus(), tokenizer, length, config.seed());
    }

    private static StatefulTrainingDataset sequentialDataset(
            DeepJPrismTrainingConfig config, BPETokenizer tokenizer,
            int length, TextFileRange range) throws IOException {
        if (range == null) return new SequentialTextDataset(config.files().corpus(), tokenizer, length);
        return new SequentialTextDataset(config.files().corpus(), tokenizer, length, range);
    }

    static BPETokenizer prepareTokenizer(DeepJPrismTrainingConfig config)
            throws IOException {
        Path path = config.files().outputDirectory().resolve(TOKENIZER_FILE);
        Files.createDirectories(path.getParent());
        BPEModel model = Files.isRegularFile(path) ? BPEModelIO.load(path) : trainTokenizer(config, path);
        validateTokenizer(model, config.tokenizer().vocabSize());
        return new BPETokenizer(model);
    }

    private static BPEModel trainTokenizer(DeepJPrismTrainingConfig config, Path path)
            throws IOException {
        System.out.printf("Training BPE tokenizer: vocab=%d sample=%d MiB%n",
                config.tokenizer().vocabSize(), config.tokenizer().sampleMiB());
        BPEModel model = new BPETrainer().trainFromFile(config.files().corpus(),
                config.tokenizer().vocabSize(), SPECIAL_TOKENS, config.tokenizer().sampleChars());
        BPEModelIO.save(path, model);
        return model;
    }

    private static void validateTokenizer(BPEModel model, int expectedVocabSize) {
        if (model.vocabSize() != expectedVocabSize) {
            throw new IllegalStateException("Tokenizer vocabulary does not match deepj.vocabSize");
        }
        for (String token : SPECIAL_TOKENS) {
            if (!model.specialTokenToId().containsKey(token)) {
                throw new IllegalStateException("Tokenizer is missing special token " + token);
            }
        }
    }

    private static TrainingResult train(DeepJPrism model, StatefulTrainingDataset dataset,
                                        DeepJPrismTrainingConfig config) throws IOException {
        DeepJPrismTrainingConfig.Training options = config.training();
        printBatchConfiguration(config);
        CosineLearningRateSchedule schedule = schedule(options);
        AdamW optimizer = AdamW.defaultAdamW(schedule.learningRate(0));
        TrainingProgress progress = loadCheckpointIfRequested(
                model, optimizer, dataset, schedule, config.files().resumeCheckpoint());
        Trainer trainer = CausalLMTraining.trainer(model, dataset, optimizer);
        Trainer.StepHook hook = checkpointHook(model, optimizer, dataset, schedule, config);
        TrainingResult result = trainer.train(options.steps(), options.batchSize(), options.logEvery(),
                0.98f, null, options.releaseEvery(), hook, progress);
        model.save(config.files().outputDirectory().resolve(FINAL_MODEL_FILE));
        printResult(result);
        return result;
    }

    private static void printBatchConfiguration(DeepJPrismTrainingConfig config) {
        int sequences = config.training().batchSize();
        long tokens = Math.multiplyExact((long) sequences, config.architecture().sequenceLength());
        String unit = sequences == 1 ? "sequence" : "sequences";
        System.out.printf("Batch: %d %s (%,d tokens per optimizer update)%n",
                sequences, unit, tokens);
    }

    private static CosineLearningRateSchedule schedule(DeepJPrismTrainingConfig.Training options) {
        return new CosineLearningRateSchedule(options.peakLearningRate(), options.minimumLearningRate(),
                options.warmupSteps(), options.steps());
    }

    private static Trainer.StepHook checkpointHook(DeepJPrism model, AdamW optimizer,
                                                    StatefulTrainingDataset dataset,
                                                    CosineLearningRateSchedule schedule,
                                                    DeepJPrismTrainingConfig config) {
        return (step, loss, ema) -> {
            int completed = step + 1;
            optimizer.setLr(schedule.learningRate(completed));
            int interval = config.training().checkpointEvery();
            if (interval > 0 && completed % interval == 0) {
                saveCheckpoint(model, optimizer, dataset, schedule, config,
                        new TrainingProgress(completed, loss, ema));
            }
        };
    }

    private static void saveCheckpoint(DeepJPrism model, AdamW optimizer,
                                       StatefulTrainingDataset dataset,
                                       CosineLearningRateSchedule schedule,
                                       DeepJPrismTrainingConfig config,
                                       TrainingProgress progress) throws IOException {
        Path output = config.files().outputDirectory();
        TrainingCheckpoint.save(model.parameters(), optimizer, dataset, progress,
                schedule, output.resolve(LATEST_TRAINING_FILE));
        model.save(output.resolve(LATEST_MODEL_FILE));
    }

    private static TrainingProgress loadCheckpointIfRequested(
            DeepJPrism model, AdamW optimizer, StatefulTrainingDataset dataset,
            CosineLearningRateSchedule schedule, Path checkpoint) throws IOException {
        if (checkpoint == null) return TrainingProgress.initial();
        if (TrainingCheckpoint.matches(checkpoint)) {
            TrainingProgress progress = TrainingCheckpoint.load(
                    model.parameters(), optimizer, dataset, schedule, checkpoint);
            System.out.println("Resumed complete training state from " + checkpoint);
            return progress;
        }
        return loadLegacyCheckpoint(model, optimizer, schedule, checkpoint);
    }

    private static TrainingProgress loadLegacyCheckpoint(DeepJPrism model, AdamW optimizer,
                                                         CosineLearningRateSchedule schedule,
                                                         Path checkpoint) throws IOException {
        model.load(checkpoint);
        int completed = Integer.getInteger("deepj.resumeStep", 0);
        optimizer.setLr(schedule.learningRate(completed));
        System.out.println("Loaded weights from " + checkpoint
                + " (optimizer state starts fresh for this run)");
        return new TrainingProgress(completed, Float.NaN, Float.NaN);
    }

    private static void validateCorpus(Path corpus) {
        if (!Files.isRegularFile(corpus)) {
            throw new IllegalArgumentException("Training corpus not found: " + corpus);
        }
    }

    private static void ensureCheckpointSpace(Path output, DeepJPrism model) throws IOException {
        long checkpointBytes = model.parameters().stream()
                .mapToLong(parameter -> (long) parameter.value.data.length * Float.BYTES + 8L).sum();
        long required = checkpointBytes * 7L + 16L * 1024L * 1024L;
        long available = Files.getFileStore(output).getUsableSpace();
        if (available < required) throw new IOException("Not enough disk space for rolling checkpoints");
    }

    private static Map<String, String> metadata(DeepJPrismTrainingConfig config,
                                                DatasetOrder order) {
        Map<String, String> values = new HashMap<>();
        addFileMetadata(values, config);
        addModelMetadata(values, config);
        TrainingMetadata.add(values, config.training());
        values.put("datasetOrder", order.sequential() ? "sequential" : "random");
        addRangeMetadata(values, order.range());
        return values;
    }

    private static void addRangeMetadata(Map<String, String> target, TextFileRange range) {
        if (range == null) return;
        target.put("corpusStartByte", Long.toString(range.startInclusive()));
        target.put("corpusEndByte", Long.toString(range.endExclusive()));
    }

    private static void addFileMetadata(Map<String, String> target,
                                        DeepJPrismTrainingConfig config) {
        target.put("corpus", config.files().corpus().toAbsolutePath().toString());
        target.put("output", config.files().outputDirectory().toAbsolutePath().toString());
        target.put("seed", Long.toString(config.seed()));
    }

    private static void addModelMetadata(Map<String, String> target,
                                         DeepJPrismTrainingConfig config) {
        DeepJPrismTrainingConfig.Architecture model = config.architecture();
        target.put("vocabSize", Integer.toString(config.tokenizer().vocabSize()));
        target.put("sequenceLength", Integer.toString(model.sequenceLength()));
        target.put("dModel", Integer.toString(model.dModel()));
        target.put("heads", Integer.toString(model.heads()));
        target.put("layers", Integer.toString(model.layers()));
        target.put("dFF", Integer.toString(model.dFF()));
        target.put("qRank", Integer.toString(model.qRank()));
        target.put("kvRank", Integer.toString(model.kvRank()));
        target.put("initScale", Float.toString(model.initScale()));
        target.put("gradClipNorm", Float.toString(model.gradClipNorm()));
        target.put("tokenizerSampleMiB", Integer.toString(config.tokenizer().sampleMiB()));
    }

    private static void printResult(TrainingResult result) {
        System.out.printf("Training complete: steps=%d loss=%.6f ema=%.6f%n",
                result.steps(), result.lastLoss(), result.emaLoss());
    }

    private record DatasetOrder(boolean sequential, TextFileRange range) {

        private static DatasetOrder random() {
            return new DatasetOrder(false, null);
        }

        private static DatasetOrder sequential(TextFileRange range) {
            return new DatasetOrder(true, range);
        }
    }
}
