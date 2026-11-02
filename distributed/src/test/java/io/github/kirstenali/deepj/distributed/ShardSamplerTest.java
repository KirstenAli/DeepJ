package io.github.kirstenali.deepj.distributed;

import java.util.HashSet;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ShardSamplerTest {
    @Test void unevenPartitionsContainEverySampleExactlyOnceForEveryEpoch() {
        for (int epoch = 0; epoch < 3; epoch++) {
            var seen = new HashSet<Integer>();
            for (int rank = 0; rank < 4; rank++) {
                ShardSampler sampler = new ShardSampler(11, rank, 4, 42);
                sampler.setEpoch(epoch);
                assertEquals(2, sampler.stepsPerEpoch(2));
                for (int index : sampler.next(100)) { assertTrue(seen.add(index)); }
                assertEquals(0, sampler.next(1).length);
            }
            assertEquals(11, seen.size());
        }
    }

    @Test void checkpointRestoresTheSameEpochCursorAndFuturePermutation() {
        ShardSampler source = new ShardSampler(17, 1, 3, 42), target = new ShardSampler(17, 1, 3, 42);
        source.setEpoch(4);
        source.next(2);
        byte[] state = source.snapshot();
        target.restore(state);
        assertArrayEquals(source.next(3), target.next(3));
        assertThrows(IllegalArgumentException.class, () -> target.restore(new byte[2]));
        assertThrows(IllegalArgumentException.class, () -> new ShardSampler(17, 0, 3, 42).validate(state));
        assertThrows(IllegalArgumentException.class, () -> target.next(0));
    }

    @Test void moreWorkersThanSamplesProduceEmptyShardsWithoutPadding() {
        ShardSampler sampler = new ShardSampler(1, 2, 3, 42);
        assertEquals(0, sampler.next(3).length);
        assertEquals(1, sampler.stepsPerEpoch(3));
        assertEquals(0, new ShardSampler(0, 0, 1, 42).stepsPerEpoch(3));
    }
}
