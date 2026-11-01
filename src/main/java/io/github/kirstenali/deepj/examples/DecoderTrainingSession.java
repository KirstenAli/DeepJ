package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
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

final class DecoderTrainingSession {

    private DecoderTrainingSession() {}

    static TrainingResult run(DecoderOnlyModel model, StatefulBatchSource dataset,
                               TrainingMetadata.Settings options, Path output, Path resume) throws IOException {
        var schedule = new CosineLearningRateSchedule(options.peakLearningRate(),
                options.minimumLearningRate(), options.warmupSteps(), options.steps());
        var optimizer = AdamW.defaultAdamW(schedule.learningRate(0));
        var run = new Run(model, dataset, options, output, schedule, optimizer);
        return run.train(resume);
    }

    static void ensureCheckpointSpace(Path output, DecoderOnlyModel model) throws IOException {
        long bytes = model.parameters().stream()
                .mapToLong(parameter -> (long) parameter.value.rawData().length * Float.BYTES + 8L).sum();
        long required = bytes * 7L + 16L * 1024L * 1024L;
        if (Files.getFileStore(output).getUsableSpace() < required) {
            throw new IOException("Not enough disk space for rolling checkpoints");
        }
    }

    private record Run(DecoderOnlyModel model, StatefulBatchSource dataset,
                       TrainingMetadata.Settings options, Path output,
                       CosineLearningRateSchedule schedule, AdamW optimizer) {

        TrainingResult train(Path resume) throws IOException {
            printBatch();
            TrainingProgress progress = load(resume);
            Trainer trainer = CausalLMTraining.trainer(model, dataset, optimizer);
            var result = trainer.train(options.steps(), options.batchSize(), options.logEvery(),
                    0.98f, null, options.releaseEvery(), this::afterStep, progress);
            model.save(output.resolve("model-final.dj"));
            System.out.printf("Training complete: steps=%d loss=%.6f ema=%.6f%n",
                    result.steps(), result.lastLoss(), result.emaLoss());
            return result;
        }

        private void printBatch() {
            int sequences = options.batchSize();
            long tokens = Math.multiplyExact((long) sequences, model.config().maxSeqLen());
            System.out.printf("Batch: %d %s (%,d tokens per optimizer update)%n",
                    sequences, sequences == 1 ? "sequence" : "sequences", tokens);
        }

        private TrainingProgress load(Path resume) throws IOException {
            if (resume == null) return TrainingProgress.initial();
            if (!TrainingCheckpoint.matches(resume)) return loadWeights(resume);
            var progress = TrainingCheckpoint.load(model, optimizer, dataset, schedule, resume);
            System.out.println("Resumed complete training state from " + resume);
            return progress;
        }

        private TrainingProgress loadWeights(Path resume) throws IOException {
            model.load(resume);
            int completed = Integer.getInteger("deepj.resumeStep", 0);
            optimizer.setLr(schedule.learningRate(completed));
            System.out.println("Loaded weights from " + resume
                    + " (optimizer state starts fresh for this run)");
            return new TrainingProgress(completed, Float.NaN, Float.NaN);
        }

        private void afterStep(int step, float loss, float ema) throws IOException {
            int completed = step + 1;
            optimizer.setLr(schedule.learningRate(completed));
            if (options.checkpointEvery() > 0 && completed % options.checkpointEvery() == 0) {
                save(new TrainingProgress(completed, loss, ema));
            }
        }

        private void save(TrainingProgress progress) throws IOException {
            TrainingCheckpoint.save(model, optimizer, dataset, progress, schedule,
                    output.resolve("training-latest.dj"));
            model.save(output.resolve("model-latest.dj"));
        }
    }
}
