package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class BufferPolicyTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(BufferPolicy.class)
    void batchNormBuffersFollowPolicyWhileWeightsAndAdamStaySynchronized(BufferPolicy policy) throws Exception {
        String mode = policy == BufferPolicy.LOCAL ? "buffers-local" : "buffers-root";
        try (var cluster = new ForkCluster("batchnorm", DType.FLOAT32, new int[]{2, 2}, 1, "cpu", mode)) {
            TrainingFixture fixture = TrainingFixture.create("batchnorm", DType.FLOAT32, 123, 0, 2);
            var trainer = new DistributedTrainer(cluster.group(), fixture.objective(), fixture.optimizer(),
                    new DistributedOptions(7, policy));
            byte[] report = TrainingReport.encode(fixture, trainer.trainStep(2));
            var reports = cluster.group().gather("test.report", report);
            verify(reports.get(0), reports.get(1), fixture, policy);
            cluster.successful();
        }
    }

    private static void verify(byte[] root, byte[] worker, TrainingFixture fixture, BufferPolicy policy) {
        byte[] buffers = Wire.encode(output -> CheckpointTensors.write(output, fixture.objective().buffers()));
        int shared = root.length - buffers.length;
        assertArrayEquals(Arrays.copyOf(root, shared), Arrays.copyOf(worker, shared));
        if (policy == BufferPolicy.LOCAL) assertFalse(Arrays.equals(root, worker));
        else assertArrayEquals(root, worker);
    }
}
