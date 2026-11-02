package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(30)
class FailureSafetyTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @ValueSource(strings = {"nonfinite", "backward-failure", "gradient-nonfinite", "wrong-layout"})
    void aRemoteFailureCannotReachTheLocalOptimizer(String mode) throws Exception {
        try (var cluster = new ForkCluster("fnn", DType.FLOAT32, new int[]{2, 2}, 1, "cpu", mode)) {
            TrainingFixture fixture = TrainingFixture.create("fnn", DType.FLOAT32, 123, 0, 2);
            var trainer = new DistributedTrainer(cluster.group(), fixture.objective(), fixture.optimizer(),
                    new DistributedOptions(7, BufferPolicy.BROADCAST_FROM_ROOT));
            var before = fixture.objective().model().parameters().stream().map(p -> p.value.toFloatArray()).toList();
            assertThrows(CollectiveException.class, () -> trainer.trainStep(2));
            assertEquals(0, fixture.optimizer().stepCount());
            for (int index = 0; index < before.size(); index++) {
                var parameter = fixture.objective().model().parameters().get(index);
                assertArrayEquals(before.get(index), parameter.value.toFloatArray());
            }
        }
    }
}
