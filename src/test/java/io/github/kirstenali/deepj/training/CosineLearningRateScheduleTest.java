package io.github.kirstenali.deepj.training;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CosineLearningRateScheduleTest {

    @Test
    void warmsUpThenDecaysToMinimum() {
        CosineLearningRateSchedule schedule = new CosineLearningRateSchedule(1.0f, 0.1f, 2, 10);
        assertEquals(0.5f, schedule.learningRate(0), 1e-7f);
        assertEquals(1.0f, schedule.learningRate(1), 1e-7f);
        assertEquals(1.0f, schedule.learningRate(2), 1e-7f);
        assertTrue(schedule.learningRate(5) < schedule.learningRate(2));
        assertEquals(0.1f, schedule.learningRate(10), 1e-7f);
    }

    @Test
    void supportsNoWarmupAndRejectsInvalidArguments() {
        CosineLearningRateSchedule schedule = new CosineLearningRateSchedule(1.0f, 0.0f, 0, 4);
        assertEquals(1.0f, schedule.learningRate(0), 1e-7f);
        assertThrows(IllegalArgumentException.class,
                () -> new CosineLearningRateSchedule(1.0f, 0.0f, 4, 4));
        assertThrows(IllegalArgumentException.class,
                () -> new CosineLearningRateSchedule(0.0f, 0.0f, 0, 4));
        assertThrows(IllegalArgumentException.class, () -> schedule.learningRate(-1));
    }
}
