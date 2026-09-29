package io.github.kirstenali.deepj.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CausalBatchesTest {

    @Test
    void createsShiftedRows() {
        Batch batch = CausalBatches.from(new int[]{1, 2, 3, 4, 5, 6, 7}, 2, 3);
        assertArrayEquals(new int[]{1, 2, 3}, batch.x()[0]);
        assertArrayEquals(new int[]{2, 3, 4}, batch.y()[0]);
        assertArrayEquals(new int[]{4, 5, 6}, batch.x()[1]);
        assertArrayEquals(new int[]{5, 6, 7}, batch.y()[1]);
    }
}
