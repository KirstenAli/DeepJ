package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicInteger;

public class TrainerTest {

    @Test
    void train_runsExactlyMaxSteps_whenNoEarlyStop() {
        AtomicInteger calls = new AtomicInteger(0);

        Trainer.StepFunction step = (batchSize) -> {
            calls.incrementAndGet();
            return 1.0f;
        };

        Trainer t = new Trainer(step);
        TrainingResult r = t.train(
                10,   // maxSteps
                4,    // batchSize
                1000, // logEvery (avoid noisy output in tests)
                0.9f,
                null  // no early stop
        );

        Assertions.assertEquals(10, calls.get());
        Assertions.assertEquals(10, r.steps());
        Assertions.assertEquals(1.0f, r.lastLoss(), 1e-12f);
        Assertions.assertEquals(1.0f, r.emaLoss(), 1e-12f);
    }

    @Test
    void rejectsInvalidArgs() {
        Trainer t = new Trainer((bs) -> 1.0f);
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(0, 1, 1, 0.9f, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 0, 1, 0.9f, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 1, 0, 0.9f, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 1, 1, 1.0f, null));
        Assertions.assertThrows(IllegalArgumentException.class, () -> t.train(1, 1, 1, 0.9f, null, -1));
    }

    @Test
    void train_releasesResourcesPeriodicallyAndFinally() {
        AtomicInteger releaseCalls = new AtomicInteger(0);
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(countingBackend(releaseCalls));

        try {
            Trainer t = new Trainer((bs) -> 1.0f);
            t.train(
                    6,
                    1,
                    1000,
                    0.9f,
                    null,
                    2
            );
        } finally {
            Tensor.setBackend(previous);
        }

        // steps: 0..5 => periodic releases at steps 2 and 4, plus one final release
        Assertions.assertEquals(3, releaseCalls.get());
    }

    @Test
    void train_whenPeriodicReleaseDisabled_stillReleasesFinally() {
        AtomicInteger releaseCalls = new AtomicInteger(0);
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(countingBackend(releaseCalls));

        try {
            Trainer t = new Trainer((bs) -> 1.0f);
            t.train(
                    4,
                    1,
                    1000,
                    0.9f,
                    null,
                    0
            );
        } finally {
            Tensor.setBackend(previous);
        }

        Assertions.assertEquals(1, releaseCalls.get());
    }

    @Test
    void train_withDefaultReleaseCadence_releasesFinallyOnly_forShortRuns() {
        AtomicInteger releaseCalls = new AtomicInteger(0);
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(countingBackend(releaseCalls));

        try {
            Trainer t = new Trainer((bs) -> 1.0f);
            t.train(
                    10,
                    1,
                    1000,
                    0.9f,
                    null
            );
        } finally {
            Tensor.setBackend(previous);
        }

        Assertions.assertEquals(1, releaseCalls.get());
    }

    @Test
    void train_wrapsStepHookExceptionWithStepInfo() {
        Trainer t = new Trainer((bs) -> 1.0f);

        RuntimeException ex = Assertions.assertThrows(RuntimeException.class, () ->
                t.train(3, 1, 1000, 0.9f, null, 0, (step, loss, ema) -> {
                    if (step == 1) throw new IllegalStateException("boom");
                })
        );

        Assertions.assertTrue(ex.getMessage().contains("Step hook failed at step 1"));
        Assertions.assertNotNull(ex.getCause());
        Assertions.assertEquals("boom", ex.getCause().getMessage());
    }

    @Test
    void train_releasesFinally_evenWhenStepHookThrows() {
        AtomicInteger releaseCalls = new AtomicInteger(0);
        TensorBackend previous = Tensor.backend();
        Tensor.setBackend(countingBackend(releaseCalls));

        try {
            Trainer t = new Trainer((bs) -> 1.0f);
            Assertions.assertThrows(RuntimeException.class, () ->
                    t.train(3, 1, 1000, 0.9f, null, 0, (step, loss, ema) -> {
                        throw new IllegalStateException("hook fail");
                    })
            );
        } finally {
            Tensor.setBackend(previous);
        }

        Assertions.assertEquals(1, releaseCalls.get());
    }

    private static TensorBackend countingBackend(AtomicInteger releaseCalls) {
        return (TensorBackend) Proxy.newProxyInstance(
                TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("releaseResources")) {
                        releaseCalls.incrementAndGet();
                        return null;
                    }
                    if (method.getDeclaringClass() == Object.class) {
                        return switch (method.getName()) {
                            case "toString" -> "CountingBackendProxy";
                            case "hashCode" -> System.identityHashCode(proxy);
                            case "equals" -> proxy == args[0];
                            default -> null;
                        };
                    }
                    throw new UnsupportedOperationException("Unexpected method call: " + method.getName());
                }
        );
    }
}
