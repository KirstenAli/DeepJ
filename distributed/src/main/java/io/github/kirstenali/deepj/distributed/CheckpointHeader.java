package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

record CheckpointHeader(byte[] cohort, DistributedStep progress, AdamW.State optimizer) {
    private static final int MAGIC = 0x444A4453;

    static void write(DataOutputStream output, DistributedTrainer trainer, AdamW.State optimizer, byte[] cohort)
            throws IOException {
        output.writeInt(MAGIC);
        output.writeInt(1);
        output.writeInt(trainer.group.worldSize());
        output.writeInt(trainer.group.rank());
        Wire.writeBytes(output, cohort);
        Wire.writeBytes(output, trainer.signature());
        output.writeLong(trainer.lastStep().step());
        output.writeLong(trainer.lastStep().samples());
        output.writeFloat(trainer.lastStep().loss());
        writeOptimizer(output, optimizer);
    }

    private static void writeOptimizer(DataOutputStream output, AdamW.State state) throws IOException {
        output.writeLong(state.step());
        output.writeFloat(state.learningRate());
        output.writeFloat(state.beta1());
        output.writeFloat(state.beta2());
        output.writeFloat(state.epsilon());
        output.writeFloat(state.weightDecay());
    }

    static CheckpointHeader read(DataInputStream input, DistributedTrainer trainer) throws IOException {
        if (input.readInt() != MAGIC || input.readInt() != 1) throw new IOException("Invalid distributed checkpoint");
        if (input.readInt() != trainer.group.worldSize() || input.readInt() != trainer.group.rank()) {
            throw new IOException("Checkpoint worker topology differs");
        }
        byte[] cohort = Wire.readBytes(input);
        if (cohort.length != 16) throw new IOException("Invalid checkpoint cohort");
        if (!Arrays.equals(Wire.readBytes(input), trainer.signature())) {
            throw new IOException("Checkpoint layout differs");
        }
        DistributedStep progress = new DistributedStep(input.readLong(), input.readLong(), input.readFloat());
        AdamW.State optimizer = new AdamW.State(input.readLong(), input.readFloat(), input.readFloat(),
                input.readFloat(), input.readFloat(), input.readFloat(), List.of(), List.of());
        validate(progress, optimizer);
        return new CheckpointHeader(cohort, progress, optimizer);
    }

    private static void validate(DistributedStep progress, AdamW.State state) throws IOException {
        if (progress.step() < 0 || progress.samples() < 0 || progress.step() != state.step()
                || progress.step() > 0 && !Float.isFinite(progress.loss())) {
            throw new IOException("Invalid checkpoint progress");
        }
        if (!Float.isFinite(state.learningRate()) || !Float.isFinite(state.weightDecay())) {
            throw new IOException("Invalid checkpoint optimizer state");
        }
    }
}
