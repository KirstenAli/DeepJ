package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CheckpointTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @MethodSource("io.github.kirstenali.deepj.distributed.DistributedParityTest#families")
    void everyFamilyResumesWeightsGradientsAllAdamStateAndDataCursorExactly(String family, DType dtype) {
        TrainingFixture fixture = TrainingFixture.create(family, dtype, 123, 0, 1);
        try (var trainer = new DistributedTrainer(new LocalProcessGroup(), fixture.objective(), fixture.optimizer())) {
            trainer.trainStep(2);
            DistributedCheckpoint.save(trainer, directory);
            byte[] expected = TrainingReport.encode(fixture, trainer.trainStep(2));
            byte[] cursor = fixture.objective().workerState().snapshot();
            DistributedCheckpoint.load(trainer, directory);
            byte[] actual = TrainingReport.encode(fixture, trainer.trainStep(2));
            assertArrayEquals(expected, actual);
            assertArrayEquals(cursor, fixture.objective().workerState().snapshot());
            assertEquals(2, trainer.lastStep().step());
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void twoProcessesRestoreTheirOwnShardStatesAndSharedUpdatesExactly(DType dtype) throws Exception {
        clusterCheckpoint(dtype, "cpu");
    }

    static void clusterCheckpoint(DType dtype, String backend) throws Exception {
        try (var cluster = new ForkCluster("prism", dtype, new int[]{2, 1}, 2, backend, "checkpoint")) {
            TrainingFixture fixture = TrainingFixture.create("prism", dtype, 123, 0, 2);
            var trainer = new DistributedTrainer(cluster.group(), fixture.objective(), fixture.optimizer(),
                    new DistributedOptions(7, BufferPolicy.BROADCAST_FROM_ROOT));
            report(cluster, fixture, trainer.trainStep(2));
            DistributedCheckpoint.save(trainer, cluster.checkpoints());
            byte[] expected = report(cluster, fixture, trainer.trainStep(2));
            DistributedCheckpoint.load(trainer, cluster.checkpoints());
            byte[] actual = report(cluster, fixture, trainer.trainStep(2));
            assertArrayEquals(expected, actual);
            cluster.successful();
        }
    }

    private static byte[] report(ForkCluster cluster, TrainingFixture fixture, float loss) {
        byte[] report = TrainingReport.encode(fixture, loss);
        for (byte[] rank : cluster.group().gather("test.report", report)) { assertArrayEquals(report, rank); }
        return report;
    }

    @Test void corruptionLeavesLiveWeightsOptimizerAndDataStateUnchanged() throws Exception {
        TrainingFixture fixture = TrainingFixture.create("fnn", DType.BFLOAT16, 123, 0, 1);
        try (var trainer = new DistributedTrainer(new LocalProcessGroup(), fixture.objective(), fixture.optimizer())) {
            float loss = trainer.trainStep(2);
            Path path = DistributedCheckpoint.save(trainer, directory);
            byte[] before = TrainingReport.encode(fixture, loss), cursor = fixture.objective().workerState().snapshot();
            byte[] bytes = Files.readAllBytes(path);
            bytes[bytes.length - 40] ^= 1;
            Files.write(path, bytes);
            assertThrows(CollectiveException.class, () -> DistributedCheckpoint.load(trainer, directory));
            assertArrayEquals(before, TrainingReport.encode(fixture, loss));
            assertArrayEquals(cursor, fixture.objective().workerState().snapshot());
        }
    }
}
