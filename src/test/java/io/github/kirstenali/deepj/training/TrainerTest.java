
package io.github.kirstenali.deepj.training;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

public class TrainerTest {

    @Test
    void train_runsExactlyMaxSteps_whenNoEarlyStop() {
        AtomicInteger calls = new AtomicInteger(0);

        Trainer.StepFunction step = (batchSize) -> {
            calls.incrementAndGet();
            return 1.0;
        };

        Trainer t = new Trainer(step);
        TrainingResult r = t.train(
                10,   // maxSteps
                4,    // batchSize
                1000, // logEvery (avoid noisy output in tests)
                0.9,
                null  // no early stop
        );

        Assertions.assertEquals(10, calls.get());
        Assertions.assertEquals(10, r.steps());
        Assertions.assertEquals(1.0, r.lastLoss(), 1e-12);
        Assertions.assertEquals(1.0, r.emaLoss(), 1e-12);
    }

    @Test
    void rejectsInvalidArgs() {
        Trainer t = new Trainer((bs) -> 1.0);
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(0, 1, 1, 0.9, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 0, 1, 0.9, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 1, 0, 0.9, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 1, 1, 1.0, null));
    }
}
