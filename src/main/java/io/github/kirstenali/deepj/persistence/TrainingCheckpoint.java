package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.TransformerConfig;
import io.github.kirstenali.deepj.models.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;
import io.github.kirstenali.deepj.training.TrainingProgress;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

public final class TrainingCheckpoint {

    private static final int MAGIC = 0x444A5443;
    private static final int VERSION = 2;
    private static final int TYPED_VERSION = 3;

    private TrainingCheckpoint() {}

    public static boolean matches(Path path) throws IOException {
        try (DataInputStream input = input(path)) {
            return input.readInt() == MAGIC;
        }
    }

    public static void save(List<Parameter> params, AdamW optimizer,
                            StatefulBatchSource dataset, TrainingProgress progress,
                            CosineLearningRateSchedule schedule, Path path) throws IOException {
        save(params, optimizer, dataset, progress, schedule, path, null);
    }

    public static void save(DecoderOnlyModel model, AdamW optimizer,
                            StatefulBatchSource dataset, TrainingProgress progress,
                            CosineLearningRateSchedule schedule, Path path) throws IOException {
        save(model.parameters(), optimizer, dataset, progress, schedule, path, model.config());
    }

    private static void save(List<Parameter> params, AdamW optimizer,
                             StatefulBatchSource dataset, TrainingProgress progress,
                             CosineLearningRateSchedule schedule, Path path,
                             TransformerConfig config) throws IOException {
        ensureParent(path);
        Path temporary = temporaryPath(path);
        try {
            write(params, optimizer.state(params), dataset.randomState(), progress,
                    schedule, temporary, config);
            replaceAtomically(temporary, path);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public static TrainingProgress load(List<Parameter> params, AdamW optimizer,
                                        StatefulBatchSource dataset,
                                        CosineLearningRateSchedule schedule,
                                        Path path) throws IOException {
        return load(params, optimizer, dataset, schedule, path, null);
    }

    public static TrainingProgress load(DecoderOnlyModel model, AdamW optimizer,
                                        StatefulBatchSource dataset, CosineLearningRateSchedule schedule,
                                        Path path) throws IOException {
        return load(model.parameters(), optimizer, dataset, schedule, path, model.config());
    }

    private static TrainingProgress load(List<Parameter> params, AdamW optimizer,
                                         StatefulBatchSource dataset, CosineLearningRateSchedule schedule,
                                         Path path, TransformerConfig expected) throws IOException {
        ModelSerializer.prepareForLoad();
        CheckpointData checkpoint = read(params, schedule, path, expected);
        optimizer.validateState(params, checkpoint.optimizer());
        TrainingTensorIO.apply(params, checkpoint.modelValues());
        optimizer.restoreState(params, checkpoint.optimizer());
        dataset.restoreRandomState(checkpoint.datasetState());
        return checkpoint.progress();
    }

    private static void write(List<Parameter> params, AdamW.State optimizer, long datasetState,
                              TrainingProgress progress, CosineLearningRateSchedule schedule,
                              Path path, TransformerConfig config) throws IOException {
        try (DataOutputStream output = output(path)) {
            boolean typed = TrainingTensorIO.requiresTyped(params, optimizer);
            writeHeader(output, progress, datasetState, config, typed);
            writeSchedule(output, schedule);
            writeOptimizer(output, optimizer);
            TrainingTensorIO.write(output, params, optimizer, typed);
        }
    }

    private static void writeHeader(DataOutputStream output, TrainingProgress progress,
                                    long datasetState, TransformerConfig config, boolean typed) throws IOException {
        output.writeInt(MAGIC);
        output.writeInt(typed ? TYPED_VERSION : config == null ? 1 : VERSION);
        if (typed) output.writeBoolean(config != null);
        if (config != null) ModelSettingsIO.write(output, new ModelSettings(config, PromptFormat.PLAIN));
        output.writeInt(progress.completedSteps());
        output.writeFloat(progress.lastLoss());
        output.writeFloat(progress.emaLoss());
        output.writeLong(datasetState);
    }

    private static void writeSchedule(DataOutputStream output,
                                      CosineLearningRateSchedule schedule) throws IOException {
        output.writeFloat(schedule.peakLearningRate());
        output.writeFloat(schedule.minimumLearningRate());
        output.writeInt(schedule.warmupSteps());
        output.writeInt(schedule.totalSteps());
    }

    private static void writeOptimizer(DataOutputStream output,
                                       AdamW.State optimizer) throws IOException {
        output.writeLong(optimizer.step());
        output.writeFloat(optimizer.learningRate());
        output.writeFloat(optimizer.beta1());
        output.writeFloat(optimizer.beta2());
        output.writeFloat(optimizer.epsilon());
        output.writeFloat(optimizer.weightDecay());
    }

    private static CheckpointData read(List<Parameter> params,
                                       CosineLearningRateSchedule schedule,
                                       Path path, TransformerConfig expected) throws IOException {
        try (var file = new CheckpointInput(path)) {
            DataInputStream input = file.data();
            Metadata metadata = readMetadata(input, schedule, expected);
            validateCount(input.readInt(), params.size());
            TrainingTensorIO.State tensors = TrainingTensorIO.read(file, params, metadata.typed());
            if (input.read() != -1) throw new IOException("Unexpected trailing checkpoint data");
            AdamW.State optimizer = metadata.optimizer(tensors);
            return new CheckpointData(metadata.progress(), metadata.datasetState(),
                    optimizer, tensors.model());
        }
    }

    private static Metadata readMetadata(DataInputStream input,
                                         CosineLearningRateSchedule schedule,
                                         TransformerConfig expected) throws IOException {
        int magic = input.readInt(), version = input.readInt();
        validateFormat(magic, version);
        boolean configured = version == TYPED_VERSION ? CheckpointTensorIO.readFlag(input) : version == VERSION;
        validateConfiguration(input, configured, expected);
        TrainingProgress progress = new TrainingProgress(
                input.readInt(), input.readFloat(), input.readFloat());
        long datasetState = input.readLong();
        validateSchedule(input, schedule);
        return readOptimizerMetadata(input, progress, datasetState, version == TYPED_VERSION);
    }

    private static Metadata readOptimizerMetadata(DataInputStream input,
                                                  TrainingProgress progress,
                                                  long datasetState, boolean typed) throws IOException {
        return new Metadata(progress, datasetState, input.readLong(), input.readFloat(),
                input.readFloat(), input.readFloat(), input.readFloat(), input.readFloat(), typed);
    }

    private static void validateFormat(int magic, int version) throws IOException {
        if (magic != MAGIC) throw new IOException("Not a DeepJ training checkpoint");
        if (version != 1 && version != VERSION && version != TYPED_VERSION) {
            throw new IOException("Unsupported training checkpoint version");
        }
    }

    private static void validateConfiguration(DataInputStream input, boolean configured,
                                              TransformerConfig expected) throws IOException {
        if (!configured) {
            if (expected instanceof DeepJLoopConfig) {
                throw new IOException("Loop checkpoint lacks architecture settings");
            }
            return;
        }
        TransformerConfig actual = ModelSettingsIO.read(input).config();
        if (!actual.equals(expected)) throw new IOException("Training configuration does not match checkpoint");
    }

    private static void validateSchedule(DataInputStream input,
                                         CosineLearningRateSchedule schedule) throws IOException {
        float peak = input.readFloat();
        float minimum = input.readFloat();
        int warmup = input.readInt();
        int total = input.readInt();
        if (Float.compare(peak, schedule.peakLearningRate()) != 0
                || Float.compare(minimum, schedule.minimumLearningRate()) != 0
                || warmup != schedule.warmupSteps() || total != schedule.totalSteps()) {
            throw new IOException("Learning-rate schedule does not match checkpoint");
        }
    }

    private static void validateCount(int count, int expected) throws IOException {
        if (count != expected) throw new IOException("Parameter count mismatch");
    }

    private static DataInputStream input(Path path) throws IOException {
        return new DataInputStream(new BufferedInputStream(Files.newInputStream(path)));
    }

    private static DataOutputStream output(Path path) throws IOException {
        return new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)));
    }

    private static void ensureParent(Path path) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static Path temporaryPath(Path path) throws IOException {
        Path absolute = path.toAbsolutePath();
        return Files.createTempFile(absolute.getParent(), absolute.getFileName().toString(), ".tmp");
    }

    private static void replaceAtomically(Path temporary, Path destination) throws IOException {
        try {
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException exception) {
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private record Metadata(TrainingProgress progress, long datasetState, long optimizerStep,
                            float learningRate, float beta1, float beta2,
                            float epsilon, float weightDecay, boolean typed) {

        AdamW.State optimizer(TrainingTensorIO.State tensors) {
            return new AdamW.State(optimizerStep, learningRate, beta1, beta2,
                    epsilon, weightDecay, tensors.first(), tensors.second(), tensors.masters());
        }
    }

    private record CheckpointData(TrainingProgress progress, long datasetState,
                                  AdamW.State optimizer, List<Tensor> modelValues) {}
}
