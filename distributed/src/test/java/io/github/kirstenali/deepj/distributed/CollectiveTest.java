package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import java.time.Duration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

@Timeout(30)
class CollectiveTest {
    @Test void threeProcessesSumBroadcastAndGatherUsingTheProductionProtocol() throws Exception {
        try (var cluster = new ForkCluster("fnn", DType.FLOAT32, new int[]{1, 1, 1}, 0, "cpu", "probe")) {
            float[] values = {0, 2};
            cluster.group().sum("probe", values);
            assertArrayEquals(new float[]{3, 6}, values);
            values[0] = 42;
            cluster.group().broadcast("broadcast", values);
            for (byte[] report : cluster.group().gather("probe.report", CollectiveValues.floats(values))) {
                float[] copy = new float[2];
                CollectiveValues.copy(report, copy);
                assertArrayEquals(values, copy);
            }
            cluster.successful();
        }
    }

    @ParameterizedTest @ValueSource(strings = {"bad-key", "bad-size", "crash", "stall"})
    void failuresPoisonTheGroupAndNeverWaitIndefinitely(String mode) throws Exception {
        try (var cluster = new ForkCluster("fnn", DType.FLOAT32, new int[]{1, 1}, 0, "cpu", mode,
                Duration.ofSeconds(2))) {
            assertThrows(CollectiveException.class, () -> cluster.group().sum("probe", new float[]{1, 2}));
            assertThrows(CollectiveException.class, () -> cluster.group().barrier("after-failure"));
        }
    }

    @Test void oneWorkerCollectivesAreIdentityOperationsAndCloseIsTerminal() {
        ProcessGroup group = new LocalProcessGroup();
        float[] values = {1, 2};
        group.sum("sum", values);
        group.broadcast("broadcast", values);
        assertArrayEquals(new float[]{1, 2}, values);
        assertArrayEquals(new byte[]{3}, group.gather("gather", new byte[]{3}).get(0));
        group.close();
        assertThrows(CollectiveException.class, () -> group.barrier("closed"));
    }
}
