package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

public final class TrainDeepSeekTinyStories {

    private static final List<String> SPECIAL_TOKENS =
            List.of("<BOS>", "<EOS>", "<PAD>", "<|endoftext|>");
    private static final String TOKENIZER_FILE = "tokenizer.bpe";
    private static final String LATEST_MODEL_FILE = "model-latest.dj";
    private static final String FINAL_MODEL_FILE = "model-final.dj";

    private TrainDeepSeekTinyStories() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        run(DeepSeekTinyStoriesConfig.fromSystemProperties());
    }

    public static TrainingResult run(DeepSeekTinyStoriesConfig runConfig) throws Exception {
        Path output = runConfig.files().outputDirectory();
        validateCorpus(runConfig.files().corpus());
        Files.createDirectories(output);
        BPETokenizer tokenizer = loadOrTrainTokenizer(runConfig);
        DeepSeekConfig modelConfig = runConfig.modelConfig(tokenizer.vocabSize());
        DeepSeekModel model = new DeepSeekModel(modelConfig, runConfig.seed());
        loadCheckpointIfRequested(model, runConfig.files().resumeCheckpoint());
        ensureCheckpointSpace(output, model);
        writeConfiguration(output, runConfig, tokenizer.model());
        try (RandomAccessTextDataset dataset = dataset(runConfig, tokenizer)) {
            return train(model, dataset, runConfig);
        }
    }

    private static RandomAccessTextDataset dataset(DeepSeekTinyStoriesConfig config,
                                                   BPETokenizer tokenizer) throws IOException {
        return new RandomAccessTextDataset(config.files().corpus(), tokenizer,
                config.architecture().sequenceLength(), config.seed());
    }

    private static BPETokenizer loadOrTrainTokenizer(DeepSeekTinyStoriesConfig config)
            throws IOException {
        Path path = config.files().outputDirectory().resolve(TOKENIZER_FILE);
        BPEModel model = Files.isRegularFile(path) ? BPEModelIO.load(path) : trainTokenizer(config, path);
        validateTokenizer(model, config.tokenizer().vocabSize());
        return new BPETokenizer(model);
    }

    private static BPEModel trainTokenizer(DeepSeekTinyStoriesConfig config, Path path)
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

    private static TrainingResult train(DeepSeekModel model, RandomAccessTextDataset dataset,
                                        DeepSeekTinyStoriesConfig config) throws IOException {
        DeepSeekTinyStoriesConfig.Training options = config.training();
        CosineLearningRateSchedule schedule = schedule(options);
        AdamW optimizer = AdamW.defaultAdamW(schedule.learningRate(0));
        Trainer trainer = CausalLMTraining.trainer(model, dataset, optimizer);
        Trainer.StepHook hook = checkpointHook(model, optimizer, schedule, config);
        TrainingResult result = trainer.train(options.steps(), options.batchSize(), options.logEvery(),
                0.98f, null, options.releaseEvery(), hook);
        model.save(config.files().outputDirectory().resolve(FINAL_MODEL_FILE));
        printResult(result);
        return result;
    }

    private static CosineLearningRateSchedule schedule(DeepSeekTinyStoriesConfig.Training options) {
        return new CosineLearningRateSchedule(options.peakLearningRate(), options.minimumLearningRate(),
                options.warmupSteps(), options.steps());
    }

    private static Trainer.StepHook checkpointHook(DeepSeekModel model, AdamW optimizer,
                                                    CosineLearningRateSchedule schedule,
                                                    DeepSeekTinyStoriesConfig config) {
        return (step, loss, ema) -> {
            int completed = step + 1;
            optimizer.setLr(schedule.learningRate(completed));
            int interval = config.training().checkpointEvery();
            if (interval > 0 && completed % interval == 0) {
                model.save(config.files().outputDirectory().resolve(LATEST_MODEL_FILE));
            }
        };
    }

    private static void loadCheckpointIfRequested(DeepSeekModel model, Path checkpoint)
            throws IOException {
        if (checkpoint == null) return;
        model.load(checkpoint);
        System.out.println("Loaded weights from " + checkpoint
                + " (optimizer state starts fresh for this run)");
    }

    private static void validateCorpus(Path corpus) {
        if (!Files.isRegularFile(corpus)) {
            throw new IllegalArgumentException("Training corpus not found: " + corpus);
        }
    }

    private static void ensureCheckpointSpace(Path output, DeepSeekModel model) throws IOException {
        long checkpointBytes = model.parameters().stream()
                .mapToLong(parameter -> (long) parameter.value.data.length * Float.BYTES + 8L).sum();
        long required = checkpointBytes * 3L + 16L * 1024L * 1024L;
        long available = Files.getFileStore(output).getUsableSpace();
        if (available < required) throw new IOException("Not enough disk space for rolling checkpoints");
    }

    private static void writeConfiguration(Path output, DeepSeekTinyStoriesConfig config,
                                           BPEModel tokenizer) throws IOException {
        Properties properties = new Properties();
        addFileProperties(properties, config);
        addModelProperties(properties, config, tokenizer);
        addTrainingProperties(properties, config);
        try (OutputStream stream = Files.newOutputStream(output.resolve("training.properties"))) {
            properties.store(stream, "DeepJ training configuration");
        }
    }

    private static void addFileProperties(Properties target, DeepSeekTinyStoriesConfig config) {
        target.setProperty("corpus", config.files().corpus().toAbsolutePath().toString());
        target.setProperty("output", config.files().outputDirectory().toAbsolutePath().toString());
        target.setProperty("seed", Long.toString(config.seed()));
    }

    private static void addModelProperties(Properties target, DeepSeekTinyStoriesConfig config,
                                           BPEModel tokenizer) {
        DeepSeekTinyStoriesConfig.Architecture model = config.architecture();
        target.setProperty("vocabSize", Integer.toString(tokenizer.vocabSize()));
        target.setProperty("sequenceLength", Integer.toString(model.sequenceLength()));
        target.setProperty("dModel", Integer.toString(model.dModel()));
        target.setProperty("heads", Integer.toString(model.heads()));
        target.setProperty("layers", Integer.toString(model.layers()));
        target.setProperty("dFF", Integer.toString(model.dFF()));
        target.setProperty("qRank", Integer.toString(model.qRank()));
        target.setProperty("kvRank", Integer.toString(model.kvRank()));
        target.setProperty("initScale", Float.toString(model.initScale()));
        target.setProperty("gradClipNorm", Float.toString(model.gradClipNorm()));
    }

    private static void addTrainingProperties(Properties target, DeepSeekTinyStoriesConfig config) {
        DeepSeekTinyStoriesConfig.Training training = config.training();
        target.setProperty("steps", Integer.toString(training.steps()));
        target.setProperty("batchSize", Integer.toString(training.batchSize()));
        target.setProperty("peakLearningRate", Float.toString(training.peakLearningRate()));
        target.setProperty("minimumLearningRate", Float.toString(training.minimumLearningRate()));
        target.setProperty("warmupSteps", Integer.toString(training.warmupSteps()));
        target.setProperty("tokenizerSampleMiB", Integer.toString(config.tokenizer().sampleMiB()));
    }

    private static void printResult(TrainingResult result) {
        System.out.printf("Training complete: steps=%d loss=%.6f ema=%.6f%n",
                result.steps(), result.lastLoss(), result.emaLoss());
    }
}
