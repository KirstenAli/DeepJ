package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.time.Duration;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.*;

class DistributedContractsTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @CsvSource({"0,0", "0,1025", "-1,2", "2,2"})
    void invalidRanksAndWorldSizesAreRejected(int rank, int workers) {
        assertThrows(IllegalArgumentException.class,
                () -> TcpSettings.of(rank, workers, "session", Duration.ofSeconds(1)));
    }

    @Test void sessionsAndTimeoutsMustBeBoundedBeforeConnecting() {
        for (String session : new String[]{"", " ", "x".repeat(4097)}) {
            assertThrows(IllegalArgumentException.class,
                    () -> TcpSettings.of(0, 1, session, Duration.ofSeconds(1)));
        }
        for (Duration timeout : new Duration[]{Duration.ZERO, Duration.ofMillis(-1),
                Duration.ofMillis((long) Integer.MAX_VALUE + 1)}) {
            assertThrows(IllegalArgumentException.class, () -> TcpSettings.of(0, 1, "session", timeout));
        }
    }

    @Test void tensorBatchesRejectScalarsEmptyBatchesAndUnequalCounts() {
        Tensor scalar = Tensor.from(new float[]{1}).reshape();
        Tensor rows = Tensor.zeros(2, 1);
        assertThrows(IllegalArgumentException.class, () -> new TensorBatch(scalar, rows));
        assertThrows(IllegalArgumentException.class, () -> new TensorBatch(rows, scalar));
        assertThrows(IllegalArgumentException.class, () -> new TensorBatch(Tensor.zeros(0, 1), rows));
        assertThrows(IllegalArgumentException.class, () -> new TensorBatch(rows, Tensor.zeros(1, 1)));
    }

    @Test void tokenShardsRequireMatchingInputTargetAndMaskCounts() {
        var sampler = new ShardSampler(1, 0, 1, 42);
        int[][] tokens = {{1}};
        for (Batch batch : new Batch[]{new Batch(null, tokens), new Batch(tokens, null),
                new Batch(new int[0][], tokens), new Batch(tokens, new int[0][]),
                new Batch(tokens, tokens, new boolean[0][])}) {
            assertThrows(IllegalArgumentException.class, () -> new CausalShard(batch, sampler));
        }
    }

    @Test void lossWeightsAndBucketSizesRejectInvalidBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> new WeightedLoss(Tensor.from(new float[]{0, 0}), 1));
        assertThrows(IllegalArgumentException.class, () -> new WeightedLoss(Tensor.from(new float[]{0}), -1));
        for (int size : new int[]{0, Wire.MAX_PAYLOAD / Float.BYTES + 1}) {
            assertThrows(IllegalArgumentException.class, () -> new DistributedOptions(size, BufferPolicy.LOCAL));
        }
        assertEquals(262144, DistributedOptions.defaults().bucketElements());
    }

    @Test void statelessWorkersAcceptOnlyAnEmptyCheckpointState() {
        WorkerState state = WorkerState.empty();
        assertArrayEquals(new byte[0], state.snapshot());
        assertDoesNotThrow(() -> state.restore(new byte[0]));
        assertThrows(IllegalArgumentException.class, () -> state.restore(new byte[]{1}));
    }
}
