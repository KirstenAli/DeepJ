package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingProgress;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Properties;

public final class FineTuneDeepSeekKnowledge {

    private static final String LATEST_MODEL = "model-latest.dj";
    private static final String LATEST_TRAINING = "training-latest.dj";
    private static final String FINAL_MODEL = "model-final.dj";

    private FineTuneDeepSeekKnowledge() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        run(KnowledgeFineTuningConfig.fromSystemProperties());
    }

    static TrainingResult run(KnowledgeFineTuningConfig config) throws Exception {
        var artifacts = DeepSeekTinyStoriesArtifacts.load(
                config.files().base(), config.files().initialModel());
        printInitialModel(config);
        prepareOutput(config, artifacts.model());
        var resources = new FineTuningResources(config, artifacts);
        return train(config, artifacts.model(), resources);
    }

    private static void printInitialModel(KnowledgeFineTuningConfig config) {
        if (config.files().resume() == null) {
            System.out.println("Loaded initial weights from " + config.files().initialModel());
        }
    }

    private static TrainingResult train(KnowledgeFineTuningConfig config, DeepSeekModel model,
                                        FineTuningResources resources) throws Exception {
        var options = config.training();
        var progress = loadProgress(config, model, resources);
        Trainer trainer = CausalLMTraining.trainer(model, resources.dataset(), resources.optimizer());
        TrainingResult result = trainer.train(options.steps(), options.batchSize(), options.logEvery(),
                0.98f, null, options.releaseEvery(), hook(config, model, resources), progress);
        model.save(config.files().output().resolve(FINAL_MODEL));
        printResult(result);
        return result;
    }

    private static TrainingProgress loadProgress(KnowledgeFineTuningConfig config,
                                                 DeepSeekModel model,
                                                 FineTuningResources resources) throws IOException {
        Path resume = config.files().resume();
        if (resume == null) return TrainingProgress.initial();
        if (!TrainingCheckpoint.matches(resume)) return loadWeights(model, resume);
        TrainingProgress progress = TrainingCheckpoint.load(model.parameters(), resources.optimizer(),
                resources.dataset(), resources.schedule(), resume);
        System.out.println("Resumed complete fine-tuning state from " + resume);
        return progress;
    }

    private static TrainingProgress loadWeights(DeepSeekModel model, Path checkpoint)
            throws IOException {
        model.load(checkpoint);
        System.out.println("Loaded initial weights from " + checkpoint);
        return TrainingProgress.initial();
    }

    private static Trainer.StepHook hook(KnowledgeFineTuningConfig config, DeepSeekModel model,
                                         FineTuningResources resources) {
        return (step, loss, ema) -> afterStep(config, model, resources,
                new TrainingProgress(step + 1, loss, ema));
    }

    private static void afterStep(KnowledgeFineTuningConfig config, DeepSeekModel model,
                                  FineTuningResources resources,
                                  TrainingProgress progress) throws IOException {
        resources.optimizer().setLr(resources.schedule().learningRate(progress.completedSteps()));
        int interval = config.training().checkpointEvery();
        if (interval > 0 && progress.completedSteps() % interval == 0) {
            saveCheckpoint(config.files().output(), model, resources, progress);
        }
    }

    private static void saveCheckpoint(Path output, DeepSeekModel model,
                                       FineTuningResources resources,
                                       TrainingProgress progress) throws IOException {
        TrainingCheckpoint.save(model.parameters(), resources.optimizer(), resources.dataset(),
                progress, resources.schedule(), output.resolve(LATEST_TRAINING));
        model.save(output.resolve(LATEST_MODEL));
    }

    private static void prepareOutput(KnowledgeFineTuningConfig config, DeepSeekModel model)
            throws IOException {
        Files.createDirectories(config.files().output());
        ensureCheckpointSpace(config.files().output(), model);
        copyBaseArtifacts(config.files());
        writeConfiguration(config);
    }

    private static void copyBaseArtifacts(KnowledgeFineTuningConfig.FilesConfig files)
            throws IOException {
        copy(files.base().resolve("tokenizer.bpe"), files.output().resolve("tokenizer.bpe"));
        copy(files.base().resolve("training.properties"),
                files.output().resolve("training.properties"));
    }

    private static void copy(Path source, Path target) throws IOException {
        if (!source.equals(target)) Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void ensureCheckpointSpace(Path output, DeepSeekModel model) throws IOException {
        long bytes = model.parameters().stream()
                .mapToLong(parameter -> (long) parameter.value.data.length * Float.BYTES + 8L).sum();
        long required = bytes * 7L + 16L * 1024L * 1024L;
        if (Files.getFileStore(output).getUsableSpace() < required) {
            throw new IOException("Not enough disk space for fine-tuning checkpoints");
        }
    }

    private static void writeConfiguration(KnowledgeFineTuningConfig config) throws IOException {
        Properties properties = baseProperties(config);
        addTrainingProperties(properties, config);
        try (OutputStream stream = Files.newOutputStream(
                config.files().output().resolve("fine-tuning.properties"))) {
            properties.store(stream, "DeepJ response-only fine-tuning configuration");
        }
    }

    private static Properties baseProperties(KnowledgeFineTuningConfig config) {
        Properties properties = new Properties();
        properties.setProperty("base", absolute(config.files().base()));
        properties.setProperty("initialModel", absolute(config.files().initialModel()));
        properties.setProperty("alpacaCorpus", absolute(config.files().alpaca()));
        properties.setProperty("factCorpus", absolute(config.files().facts()));
        properties.setProperty("seed", Long.toString(config.seed()));
        return properties;
    }

    private static void addTrainingProperties(Properties target,
                                              KnowledgeFineTuningConfig config) {
        var training = config.training();
        target.setProperty("steps", Integer.toString(training.steps()));
        target.setProperty("batchSize", Integer.toString(training.batchSize()));
        target.setProperty("peakLearningRate", Float.toString(training.peakLearningRate()));
        target.setProperty("minimumLearningRate", Float.toString(training.minimumLearningRate()));
        target.setProperty("warmupSteps", Integer.toString(training.warmupSteps()));
        target.setProperty("checkpointEvery", Integer.toString(training.checkpointEvery()));
        target.setProperty("releaseEvery", Integer.toString(training.releaseEvery()));
        target.setProperty("alpacaWeight", Integer.toString(config.alpacaWeight()));
        target.setProperty("factWeight", Integer.toString(config.factWeight()));
    }

    private static String absolute(Path path) {
        return path.toAbsolutePath().toString();
    }

    private static void printResult(TrainingResult result) {
        System.out.printf("Fine-tuning complete: steps=%d loss=%.6f ema=%.6f%n",
                result.steps(), result.lastLoss(), result.emaLoss());
    }

    private record FineTuningResources(ResponseOnlyTextDataset dataset, AdamW optimizer,
                                       CosineLearningRateSchedule schedule) {

        private FineTuningResources(KnowledgeFineTuningConfig config,
                                    DeepSeekTinyStoriesArtifacts.Loaded artifacts)
                throws IOException {
            this(dataset(config, artifacts), optimizer(config), schedule(config));
            System.out.printf("Loaded %,d response-only records%n", dataset.examples().size());
        }

        private static ResponseOnlyTextDataset dataset(KnowledgeFineTuningConfig config,
                                                        DeepSeekTinyStoriesArtifacts.Loaded artifacts)
                throws IOException {
            var files = config.files();
            var sources = List.of(new ResponseOnlyTextDataset.Source(files.alpaca(), config.alpacaWeight()),
                    new ResponseOnlyTextDataset.Source(files.facts(), config.factWeight()));
            return new ResponseOnlyTextDataset(sources, artifacts.tokenizer(),
                    artifacts.config().maxSeqLen(), config.seed());
        }

        private static AdamW optimizer(KnowledgeFineTuningConfig config) {
            return AdamW.defaultAdamW(schedule(config).learningRate(0));
        }

        private static CosineLearningRateSchedule schedule(KnowledgeFineTuningConfig config) {
            var training = config.training();
            return new CosineLearningRateSchedule(training.peakLearningRate(),
                    training.minimumLearningRate(), training.warmupSteps(), training.steps());
        }
    }
}
