package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.data.StatefulBatchSource;
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
import java.util.ArrayList;
import java.util.List;

public final class TrainingCheckpoint {

    private static final int MAGIC = 0x444A5443;
    private static final int VERSION = 1;

    private TrainingCheckpoint() {}

    public static boolean matches(Path path) throws IOException {
        try (DataInputStream input = input(path)) {
            return input.readInt() == MAGIC;
        }
    }

    public static void save(List<Parameter> params, AdamW optimizer,
                            StatefulBatchSource dataset, TrainingProgress progress,
                            CosineLearningRateSchedule schedule, Path path) throws IOException {
        ensureParent(path);
        Path temporary = temporaryPath(path);
        try {
            write(params, optimizer.state(params), dataset.randomState(), progress,
                    schedule, temporary);
            replaceAtomically(temporary, path);
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public static TrainingProgress load(List<Parameter> params, AdamW optimizer,
                                        StatefulBatchSource dataset,
                                        CosineLearningRateSchedule schedule,
                                        Path path) throws IOException {
        ModelSerializer.prepareForLoad();
        CheckpointData checkpoint = read(params, schedule, path);
        applyModelValues(params, checkpoint.modelValues());
        optimizer.restoreState(params, checkpoint.optimizer());
        dataset.restoreRandomState(checkpoint.datasetState());
        return checkpoint.progress();
    }

    private static void write(List<Parameter> params, AdamW.State optimizer, long datasetState,
                              TrainingProgress progress, CosineLearningRateSchedule schedule,
                              Path path) throws IOException {
        try (DataOutputStream output = output(path)) {
            writeHeader(output, progress, datasetState);
            writeSchedule(output, schedule);
            writeOptimizer(output, optimizer);
            writeTensors(output, params, optimizer);
        }
    }

    private static void writeHeader(DataOutputStream output, TrainingProgress progress,
                                    long datasetState) throws IOException {
        output.writeInt(MAGIC);
        output.writeInt(VERSION);
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

    private static void writeTensors(DataOutputStream output, List<Parameter> params,
                                     AdamW.State optimizer) throws IOException {
        output.writeInt(params.size());
        for (int index = 0; index < params.size(); index++) {
            ModelSerializer.writeTensor(output, params.get(index).value);
            ModelSerializer.writeTensor(output, optimizer.firstMoments().get(index));
            ModelSerializer.writeTensor(output, optimizer.secondMoments().get(index));
        }
    }

    private static CheckpointData read(List<Parameter> params,
                                       CosineLearningRateSchedule schedule,
                                       Path path) throws IOException {
        try (DataInputStream input = input(path)) {
            Metadata metadata = readMetadata(input, schedule);
            validateCount(input.readInt(), params.size());
            TensorLists tensors = readTensors(input, params);
            if (input.read() != -1) throw new IOException("Unexpected trailing checkpoint data");
            AdamW.State optimizer = metadata.optimizer(tensors.first(), tensors.second());
            return new CheckpointData(metadata.progress(), metadata.datasetState(),
                    optimizer, tensors.model());
        }
    }

    private static Metadata readMetadata(DataInputStream input,
                                         CosineLearningRateSchedule schedule) throws IOException {
        validateFormat(input.readInt(), input.readInt());
        TrainingProgress progress = new TrainingProgress(
                input.readInt(), input.readFloat(), input.readFloat());
        long datasetState = input.readLong();
        validateSchedule(input, schedule);
        return readOptimizerMetadata(input, progress, datasetState);
    }

    private static Metadata readOptimizerMetadata(DataInputStream input,
                                                  TrainingProgress progress,
                                                  long datasetState) throws IOException {
        return new Metadata(progress, datasetState, input.readLong(), input.readFloat(),
                input.readFloat(), input.readFloat(), input.readFloat(), input.readFloat());
    }

    private static void validateFormat(int magic, int version) throws IOException {
        if (magic != MAGIC) throw new IOException("Not a DeepJ training checkpoint");
        if (version != VERSION) throw new IOException("Unsupported training checkpoint version");
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

    private static TensorLists readTensors(DataInputStream input,
                                           List<Parameter> params) throws IOException {
        List<Tensor> model = new ArrayList<>(params.size());
        List<Tensor> first = new ArrayList<>(params.size());
        List<Tensor> second = new ArrayList<>(params.size());
        for (int index = 0; index < params.size(); index++) {
            Tensor template = params.get(index).value;
            model.add(readTensor(input, template, index));
            first.add(readTensor(input, template, index));
            second.add(readTensor(input, template, index));
        }
        return new TensorLists(model, first, second);
    }

    private static Tensor readTensor(DataInputStream input, Tensor template,
                                     int index) throws IOException {
        Tensor tensor = new Tensor(template.rows, template.cols);
        ModelSerializer.readTensor(input, tensor, index);
        return tensor;
    }

    private static void applyModelValues(List<Parameter> params, List<Tensor> values) {
        for (int index = 0; index < params.size(); index++) {
            Tensor target = params.get(index).value;
            System.arraycopy(values.get(index).data, 0, target.data, 0, target.data.length);
            ModelSerializer.markGpuBufferNeedsUpload(target);
        }
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
                            float epsilon, float weightDecay) {

        AdamW.State optimizer(List<Tensor> first, List<Tensor> second) {
            return new AdamW.State(optimizerStep, learningRate, beta1, beta2,
                    epsilon, weightDecay, first, second);
        }
    }

    private record TensorLists(List<Tensor> model, List<Tensor> first, List<Tensor> second) {}

    private record CheckpointData(TrainingProgress progress, long datasetState,
                                  AdamW.State optimizer, List<Tensor> modelValues) {}
}
