package io.github.kirstenali.deepj.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StatefulRandomTest {

    @Test
    void restoredStateContinuesTheSameSequence() {
        StatefulRandom random = new StatefulRandom(42L);
        for (int index = 0; index < 5; index++) random.nextLong(97L);
        long state = random.state();
        long expected = random.nextLong(97L);
        StatefulRandom restored = new StatefulRandom(0L);
        restored.restore(state);
        assertEquals(expected, restored.nextLong(97L));
    }

    @Test
    void valuesStayWithinTheRequestedBound() {
        StatefulRandom random = new StatefulRandom(7L);
        for (int index = 0; index < 1_000; index++) {
            long value = random.nextLong(13L);
            assertTrue(value >= 0 && value < 13L);
        }
    }

    @Test
    void rejectsNonPositiveBounds() {
        StatefulRandom random = new StatefulRandom(1L);
        assertThrows(IllegalArgumentException.class, () -> random.nextLong(0L));
    }
}
