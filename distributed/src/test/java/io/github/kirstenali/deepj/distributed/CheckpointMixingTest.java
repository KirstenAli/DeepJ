package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(30)
class CheckpointMixingTest {
    @Test void sameStepFilesFromDifferentSavesAreRejectedBeforeAnyLiveRestore() throws Exception {
        Tensor.setBackend(new CpuBackend());
        try (var cluster = new ForkCluster("fnn", DType.FLOAT32, new int[]{2, 1}, 2, "cpu", "checkpoint-mixed")) {
            TrainingFixture fixture = TrainingFixture.create("fnn", DType.FLOAT32, 123, 0, 2);
            var trainer = new DistributedTrainer(cluster.group(), fixture.objective(), fixture.optimizer(),
                    new DistributedOptions(7, BufferPolicy.BROADCAST_FROM_ROOT));
            report(cluster, fixture, trainer.trainStep(2));
            DistributedCheckpoint.save(trainer, cluster.checkpoints());
            float loss = trainer.trainStep(2);
            byte[] before = report(cluster, fixture, loss), cursor = fixture.objective().workerState().snapshot();
            cluster.group().barrier("test.mixed.ready");
            assertThrows(CollectiveException.class, () -> DistributedCheckpoint.load(trainer, cluster.checkpoints()));
            assertArrayEquals(before, TrainingReport.encode(fixture, loss));
            assertArrayEquals(cursor, fixture.objective().workerState().snapshot());
        }
    }

    private static byte[] report(ForkCluster cluster, TrainingFixture fixture, float loss) {
        byte[] bytes = TrainingReport.encode(fixture, loss);
        for (byte[] rank : cluster.group().gather("test.report", bytes)) { assertArrayEquals(bytes, rank); }
        return bytes;
    }

    /** Change the save identifier while retaining a valid checksum, as with an intact file from another save. */
    static void relabel(Path directory, int rank) throws Exception {
        Path file = DistributedCheckpoint.path(directory, rank);
        byte[] bytes = Files.readAllBytes(file);
        bytes[20] ^= 1;
        byte[] checksum = MessageDigest.getInstance("SHA-256").digest(Arrays.copyOf(bytes, bytes.length - 32));
        System.arraycopy(checksum, 0, bytes, bytes.length - 32, checksum.length);
        Files.write(file, bytes);
    }
}
