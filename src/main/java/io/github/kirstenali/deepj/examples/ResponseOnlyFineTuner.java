package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.IndexedResponseTextDataset;
import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.TrainingCheckpoint;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.Trainer;
import io.github.kirstenali.deepj.training.TrainingProgress;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;

final class ResponseOnlyFineTuner {

    private static final long INDEXED_DATASET_THRESHOLD = 256L * 1024L * 1024L;
    private static final String LATEST_MODEL = "model-latest.dj";
    private static final String LATEST_TRAINING = "training-latest.dj";
    private static final String FINAL_MODEL = "model-final.dj";

    private ResponseOnlyFineTuner() {}

    static TrainingResult run(ResponseFineTuningConfig config) throws Exception {
        var artifacts = DecoderArtifacts.load(
                config.files().base(), config.files().initialModel());
        printInitialModel(config);
        prepareOutput(config, artifacts.model());
        try (var resources = new Resources(config, artifacts)) {
            return train(config, artifacts.model(), resources);
        }
    }

    private static TrainingResult train(ResponseFineTuningConfig config, DecoderOnlyModel model,
                                        Resources resources) throws Exception {
        var options = config.training();
        var progress = loadProgress(config, model, resources);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        Trainer trainer = CausalLMTraining.trainer(model, resources.dataset(), resources.optimizer());
        TrainingResult result = trainer.train(options.steps(), options.batchSize(), options.logEvery(),
                0.98f, null, options.releaseEvery(), hook(config, model, resources), progress);
        model.save(config.files().output().resolve(FINAL_MODEL));
        printResult(result);
        return result;
    }

    private static TrainingProgress loadProgress(ResponseFineTuningConfig config,
                                                 DecoderOnlyModel model, Resources resources)
            throws IOException {
        Path resume = config.files().resume();
        if (resume == null) return TrainingProgress.initial();
        if (!TrainingCheckpoint.matches(resume)) return loadWeights(model, resume);
        TrainingProgress progress = TrainingCheckpoint.load(model.parameters(), resources.optimizer(),
                resources.dataset(), resources.schedule(), resume);
        System.out.println("Resumed complete fine-tuning state from " + resume);
        return progress;
    }

    private static TrainingProgress loadWeights(DecoderOnlyModel model, Path checkpoint)
            throws IOException {
        model.load(checkpoint);
        System.out.println("Loaded initial weights from " + checkpoint);
        return TrainingProgress.initial();
    }

    private static Trainer.StepHook hook(ResponseFineTuningConfig config, DecoderOnlyModel model,
                                         Resources resources) {
        return (step, loss, ema) -> afterStep(config, model, resources,
                new TrainingProgress(step + 1, loss, ema));
    }

    private static void afterStep(ResponseFineTuningConfig config, DecoderOnlyModel model,
                                  Resources resources, TrainingProgress progress)
            throws IOException {
        resources.optimizer().setLr(resources.schedule().learningRate(progress.completedSteps()));
        int interval = config.training().checkpointEvery();
        if (interval > 0 && progress.completedSteps() % interval == 0) {
            saveCheckpoint(config.files().output(), model, resources, progress);
        }
    }

    private static void saveCheckpoint(Path output, DecoderOnlyModel model, Resources resources,
                                       TrainingProgress progress) throws IOException {
        TrainingCheckpoint.save(model.parameters(), resources.optimizer(), resources.dataset(),
                progress, resources.schedule(), output.resolve(LATEST_TRAINING));
        model.save(output.resolve(LATEST_MODEL));
    }

    private static void prepareOutput(ResponseFineTuningConfig config, DecoderOnlyModel model)
            throws IOException {
        Files.createDirectories(config.files().output());
        ensureCheckpointSpace(config.files().output(), model);
        copy(config.files().base().resolve("tokenizer.bpe"),
                config.files().output().resolve("tokenizer.bpe"));
        model.setMetadata(metadata(config, model));
    }

    private static void copy(Path source, Path target) throws IOException {
        if (!source.equals(target)) Files.copy(source, target, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void ensureCheckpointSpace(Path output, DecoderOnlyModel model) throws IOException {
        long bytes = model.parameters().stream()
                .mapToLong(parameter -> (long) parameter.value.data.length * Float.BYTES + 8L).sum();
        long required = bytes * 7L + 16L * 1024L * 1024L;
        if (Files.getFileStore(output).getUsableSpace() < required) {
            throw new IOException("Not enough disk space for fine-tuning checkpoints");
        }
    }

    private static Map<String, String> metadata(ResponseFineTuningConfig config,
                                                DecoderOnlyModel model) {
        Map<String, String> values = new HashMap<>(model.metadata());
        Map<String, String> fineTuning = new HashMap<>();
        addSourceMetadata(fineTuning, config);
        TrainingMetadata.add(fineTuning, config.training());
        fineTuning.put("base", absolute(config.files().base()));
        fineTuning.put("initialModel", absolute(config.files().initialModel()));
        fineTuning.put("seed", Long.toString(config.seed()));
        fineTuning.forEach((key, value) -> values.put("fineTuning." + key, value));
        return values;
    }

    private static void addSourceMetadata(Map<String, String> target,
                                          ResponseFineTuningConfig config) {
        target.put("sourceCount", Integer.toString(config.sources().size()));
        for (int index = 0; index < config.sources().size(); index++) {
            var source = config.sources().get(index);
            target.put("source." + index + ".path", absolute(source.path()));
            target.put("source." + index + ".weight", Integer.toString(source.weight()));
        }
    }

    private static String absolute(Path path) {
        return path.toAbsolutePath().toString();
    }

    private static void printInitialModel(ResponseFineTuningConfig config) {
        if (config.files().resume() == null) {
            System.out.println("Loaded initial weights from " + config.files().initialModel());
        }
    }

    private static void printResult(TrainingResult result) {
        System.out.printf("Fine-tuning complete: steps=%d loss=%.6f ema=%.6f%n",
                result.steps(), result.lastLoss(), result.emaLoss());
    }

    private record Resources(StatefulBatchSource dataset, AdamW optimizer,
                             CosineLearningRateSchedule schedule,
                             AutoCloseable closeable) implements AutoCloseable {

        private Resources(ResponseFineTuningConfig config,
                          DecoderArtifacts.Loaded artifacts) throws IOException {
            this(dataset(config, artifacts), optimizer(config), schedule(config));
        }

        private Resources(DatasetHandle handle, AdamW optimizer,
                          CosineLearningRateSchedule schedule) {
            this(handle.dataset(), optimizer, schedule, handle.closeable());
            System.out.printf("Loaded %,d response-only records%n", handle.records());
        }

        private static DatasetHandle dataset(ResponseFineTuningConfig config,
                                             DecoderArtifacts.Loaded artifacts)
                throws IOException {
            long bytes = sourceBytes(config);
            if (bytes >= INDEXED_DATASET_THRESHOLD) return indexed(config, artifacts);
            var dataset = new ResponseOnlyTextDataset(config.sources(), artifacts.tokenizer(),
                    artifacts.config().maxSeqLen(), config.seed());
            return new DatasetHandle(dataset, dataset.examples().size(), () -> {});
        }

        private static long sourceBytes(ResponseFineTuningConfig config) throws IOException {
            long bytes = 0;
            for (var source : config.sources()) {
                bytes = Math.addExact(bytes, Files.size(source.path()));
            }
            return bytes;
        }

        private static DatasetHandle indexed(ResponseFineTuningConfig config,
                                             DecoderArtifacts.Loaded artifacts)
                throws IOException {
            var dataset = new IndexedResponseTextDataset(config.sources(), artifacts.tokenizer(),
                    artifacts.config().maxSeqLen(), config.seed());
            return new DatasetHandle(dataset, dataset.recordCount(), dataset);
        }

        private static AdamW optimizer(ResponseFineTuningConfig config) {
            return AdamW.defaultAdamW(schedule(config).learningRate(0));
        }

        private static CosineLearningRateSchedule schedule(ResponseFineTuningConfig config) {
            var training = config.training();
            return new CosineLearningRateSchedule(training.peakLearningRate(),
                    training.minimumLearningRate(), training.warmupSteps(), training.steps());
        }

        @Override
        public void close() throws Exception {
            closeable.close();
        }
    }

    private record DatasetHandle(StatefulBatchSource dataset, long records,
                                 AutoCloseable closeable) {}
}
