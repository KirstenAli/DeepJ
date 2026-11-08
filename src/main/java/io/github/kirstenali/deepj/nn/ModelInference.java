package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.function.Supplier;

/** Captures bounded inference work while preserving device intermediates until the caller needs its output. */
public final class ModelInference {
    private static final ThreadLocal<Boolean> BATCHED = new ThreadLocal<>();

    private ModelInference() {}

    /** Completes one bounded inference block while keeping its result on the selected backend. */
    public static Tensor evaluate(Supplier<Tensor> operation) {
        if (GradMode.isEnabled()) return operation.get();
        Tensor output = batch(operation);
        if (!active()) {
            try (var fixed = ModelMemory.retain(output)) { Tensor.backend().releaseTemporaryResources(); }
        }
        return output;
    }

    public static Tensor batch(Supplier<Tensor> operation) {
        if (GradMode.isEnabled()) throw new IllegalStateException("Batched inference requires noGrad");
        boolean previous = active();
        BATCHED.set(true);
        try (var buffers = BackwardBuffers.begin()) {
            Tensor output = buffers.capture(operation);
            buffers.expose(output);
            return output;
        } finally {
            if (previous) BATCHED.set(true);
            else BATCHED.remove();
        }
    }

    /** Whether this thread is currently capturing a bounded inference block. */
    public static boolean active() { return Boolean.TRUE.equals(BATCHED.get()); }
}
