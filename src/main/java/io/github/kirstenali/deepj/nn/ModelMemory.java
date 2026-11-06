package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;
import java.util.Objects;

public final class ModelMemory {
    private ModelMemory() {}

    public static void checkCancelled() {
        if (Thread.currentThread().isInterrupted()) {
            throw new java.util.concurrent.CancellationException("Generation stopped");
        }
    }

    public static Pin retain(Tensor... tensors) { return new Pin(tensors); }

    public static float[] finite(Tensor tensor, String stage) {
        float[] values = tensor.toFloatArray();
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite " + stage + " output");
        }
        return values;
    }

    public static final class Pin implements AutoCloseable {
        private final Tensor[] tensors;
        private Pin(Tensor[] values) {
            Arrays.stream(values).forEach(Objects::requireNonNull);
            tensors = new Tensor[values.length];
            try (var scope = TensorScope.open()) {
                for (int i = 0; i < values.length; i++) {
                    tensors[i] = Tensor.backend().expand(values[i], values[i].shape()).retainDeviceBuffer();
                }
            } catch (RuntimeException | Error failure) {
                close();
                throw failure;
            }
        }
        @Override public void close() {
            for (Tensor tensor : tensors) {
                if (tensor != null) tensor.releaseDeviceBufferRetention();
            }
        }
    }

    public static Tensor finish(Tensor output) {
        if (!GradMode.isEnabled()) {
            output.materialize();
            try (var pinned = retain(output)) { Tensor.backend().releaseTemporaryResources(); }
        }
        return output;
    }
}
