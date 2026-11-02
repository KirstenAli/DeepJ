package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

/** Each rank keeps one atomic local file; a successful save requires every worker to complete the collective. */
public final class DistributedCheckpoint {
    private DistributedCheckpoint() {}

    public static Path save(DistributedTrainer trainer, Path directory) {
        synchronized (trainer) {
            trainer.initialize();
            agreeOnStep(trainer, trainer.lastStep());
            Path temporary = prepare(trainer, directory, CheckpointCohort.create(trainer.group));
            return commit(trainer, directory, temporary);
        }
    }

    private static Path prepare(DistributedTrainer trainer, Path directory, byte[] cohort) {
        Path temporary = null;
        Exception failure = null;
        try {
            Files.createDirectories(directory);
            temporary = Files.createTempFile(directory, ".deepj-", ".tmp");
            CheckpointCodec.write(temporary, trainer, adam(trainer), cohort);
        } catch (IOException | RuntimeException error) { failure = error; }
        try { RankAgreement.success(trainer.group, "checkpoint.prepare", failure); }
        catch (RuntimeException error) {
            delete(temporary);
            throw error;
        }
        return temporary;
    }

    private static Path commit(DistributedTrainer trainer, Path directory, Path temporary) {
        Path target = path(directory, trainer.group.rank());
        Exception failure = null;
        try { replace(temporary, target); }
        catch (IOException error) { failure = error; }
        finally { delete(temporary); }
        RankAgreement.success(trainer.group, "checkpoint.commit", failure);
        return target;
    }

    public static DistributedStep load(DistributedTrainer trainer, Path directory) {
        synchronized (trainer) {
            trainer.initialize();
            try (var scope = TensorScope.open()) {
                CheckpointImage image = validate(trainer, path(directory, trainer.group.rank()));
                RankAgreement.same(trainer.group, "checkpoint.cohort", image.cohort());
                agreeOnStep(trainer, image.progress());
                apply(trainer, image);
                return image.progress();
            }
        }
    }

    private static CheckpointImage validate(DistributedTrainer trainer, Path path) {
        CheckpointImage image = null;
        Exception failure = null;
        try { image = CheckpointCodec.read(path, trainer, adam(trainer)); }
        catch (IOException | RuntimeException error) { failure = error; }
        RankAgreement.success(trainer.group, "checkpoint.validate", failure);
        return image;
    }

    private static void apply(DistributedTrainer trainer, CheckpointImage image) {
        List<Tensor> targets = trainer.layout.parameters.stream().map(parameter -> parameter.value).toList();
        restore(targets, image.parameters());
        restore(trainer.layout.buffers, image.buffers());
        adam(trainer).restoreState(trainer.layout.parameters, image.optimizer());
        trainer.objective.workerState().restore(image.workerState());
        trainer.objective.model().zeroGrad();
        trainer.restoreProgress(image.progress());
        trainer.group.barrier("checkpoint.restored");
    }

    private static void restore(List<Tensor> targets, List<Tensor> values) {
        for (int index = 0; index < targets.size(); index++) {
            TensorEncoding.restore(targets.get(index), values.get(index));
        }
    }

    private static void agreeOnStep(DistributedTrainer trainer, DistributedStep step) {
        RankAgreement.same(trainer.group, "checkpoint.progress", Wire.encode(output -> {
            output.writeLong(step.step());
            output.writeLong(step.samples());
            output.writeFloat(step.loss());
        }));
    }

    private static AdamW adam(DistributedTrainer trainer) {
        if (!(trainer.optimizer instanceof AdamW optimizer)) {
            throw new IllegalArgumentException("Checkpointing requires DeepJ AdamW");
        }
        return optimizer;
    }

    public static Path path(Path directory, int rank) { return directory.resolve("rank-" + rank + ".djdist"); }

    private static void replace(Path temporary, Path target) throws IOException {
        try { Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
        catch (AtomicMoveNotSupportedException failure) {
            Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void delete(Path path) {
        if (path == null) return;
        try { Files.deleteIfExists(path); }
        catch (IOException ignored) { /* Keep the original training or collective failure. */ }
    }
}
