package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

/** Validates the entire file and checksum into independent tensors before any live restoration. */
final class CheckpointCodec {
    private CheckpointCodec() {}

    static void write(Path path, DistributedTrainer trainer, AdamW optimizer, byte[] cohort) throws IOException {
        AdamW.State state = optimizer.state(trainer.layout.parameters);
        if (state.step() != trainer.lastStep().step()) throw new IOException("Optimizer advanced outside this trainer");
        var digest = new DigestOutputStream(new BufferedOutputStream(Files.newOutputStream(path)), sha256());
        try (var output = new DataOutputStream(digest)) {
            CheckpointHeader.write(output, trainer, state, cohort);
            writeTensors(output, trainer, state);
            Wire.writeBytes(output, trainer.objective.workerState().snapshot());
            output.flush();
            byte[] checksum = digest.getMessageDigest().digest();
            digest.on(false);
            output.write(checksum);
        }
    }

    private static void writeTensors(DataOutputStream output, DistributedTrainer trainer, AdamW.State state)
            throws IOException {
        CheckpointTensors.write(output, values(trainer));
        CheckpointTensors.write(output, trainer.layout.buffers);
        CheckpointTensors.write(output, state.firstMoments());
        CheckpointTensors.write(output, state.secondMoments());
        CheckpointTensors.writeMasters(output, state.masterWeights());
    }

    static CheckpointImage read(Path path, DistributedTrainer trainer, AdamW optimizer) throws IOException {
        var digest = new DigestInputStream(new BufferedInputStream(Files.newInputStream(path)), sha256());
        try (var input = new DataInputStream(digest)) {
            CheckpointHeader header = CheckpointHeader.read(input, trainer);
            CheckpointImage image = readTensors(input, trainer, header);
            byte[] checksum = digest.getMessageDigest().digest();
            digest.on(false);
            byte[] expected = new byte[32];
            input.readFully(expected);
            if (input.read() != -1 || !MessageDigest.isEqual(checksum, expected)) {
                throw new IOException("Checkpoint checksum or end-of-file mismatch");
            }
            optimizer.validateState(trainer.layout.parameters, image.optimizer());
            trainer.objective.workerState().validate(image.workerState());
            return image;
        }
    }

    private static CheckpointImage readTensors(DataInputStream input, DistributedTrainer trainer,
                                               CheckpointHeader header) throws IOException {
        List<Tensor> parameters = CheckpointTensors.read(input, values(trainer), false);
        List<Tensor> buffers = CheckpointTensors.read(input, trainer.layout.buffers, false);
        List<Tensor> first = CheckpointTensors.read(input, parameters, true);
        List<Tensor> second = CheckpointTensors.read(input, parameters, true);
        var masters = CheckpointTensors.readMasters(input, parameters);
        AdamW.State state = header.optimizer();
        var restored = new AdamW.State(state.step(), state.learningRate(), state.beta1(), state.beta2(),
                state.epsilon(), state.weightDecay(), first, second, masters);
        return new CheckpointImage(header.cohort(), header.progress(), parameters, buffers,
                restored, Wire.readBytes(input));
    }

    private static List<Tensor> values(DistributedTrainer trainer) {
        return trainer.layout.parameters.stream().map(parameter -> parameter.value).toList();
    }

    private static MessageDigest sha256() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException failure) { throw new IllegalStateException(failure); }
    }
}
