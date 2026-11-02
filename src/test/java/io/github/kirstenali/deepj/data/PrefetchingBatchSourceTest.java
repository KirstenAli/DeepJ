package io.github.kirstenali.deepj.data;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class PrefetchingBatchSourceTest {

    @Test
    void deliversBatchesInSourceOrder() {
        AtomicInteger counter = new AtomicInteger();
        try (var prefetch = new PrefetchingBatchSource(size -> batch(counter.getAndIncrement()), 1, 4)) {
            for (int expected = 0; expected < 20; expected++) {
                assertEquals(expected, prefetch.nextBatch(1).x()[0][0]);
            }
        }
    }

    @Test
    void propagatesSourceFailures() {
        BatchSource failing = size -> { throw new IllegalStateException("boom"); };
        try (var prefetch = new PrefetchingBatchSource(failing, 1, 2)) {
            IllegalStateException error = assertThrows(IllegalStateException.class, () -> prefetch.nextBatch(1));
            assertEquals("boom", error.getMessage());
        }
    }

    @Test
    void rejectsMismatchedBatchSizeAndUseAfterClose() {
        PrefetchingBatchSource prefetch = new PrefetchingBatchSource(size -> batch(0), 2, 1);
        assertThrows(IllegalArgumentException.class, () -> prefetch.nextBatch(1));
        prefetch.close();
        assertThrows(IllegalStateException.class, () -> prefetch.nextBatch(2));
    }

    @Test
    void rejectsInvalidConstruction() {
        assertThrows(IllegalArgumentException.class, () -> new PrefetchingBatchSource(null, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new PrefetchingBatchSource(size -> batch(0), 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new PrefetchingBatchSource(size -> batch(0), 1, 0));
    }

    private static Batch batch(int value) {
        return new Batch(new int[][]{{value}}, new int[][]{{value}}, null);
    }
}

