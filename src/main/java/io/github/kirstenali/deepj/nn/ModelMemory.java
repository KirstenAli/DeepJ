package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;

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
        private Pin(Tensor[] tensors) {
            this.tensors = tensors;
            for (Tensor tensor : tensors) { tensor.retainDeviceBuffer(); }
        }
        @Override public void close() {
            for (Tensor tensor : tensors) { tensor.releaseDeviceBufferRetention(); }
        }
    }

    public static Tensor finish(Tensor output) {
        if (!GradMode.isEnabled()) {
            output.materialize();
            output.retainDeviceBuffer();
            try { Tensor.backend().releaseTemporaryResources(); }
            finally { output.releaseDeviceBufferRetention(); }
        }
        return output;
    }
}
