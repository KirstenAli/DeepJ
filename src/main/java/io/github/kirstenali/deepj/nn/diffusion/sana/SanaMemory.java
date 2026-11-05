package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;

final class SanaMemory {
    private SanaMemory() {}

    static void checkCancelled() {
        if (Thread.currentThread().isInterrupted()) {
            throw new java.util.concurrent.CancellationException("Image generation stopped");
        }
    }

    static Pin retain(Tensor... tensors) { return new Pin(tensors); }

    static float[] finite(Tensor tensor, String stage) {
        float[] values = tensor.toFloatArray();
        for (float value : values) {
            if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite " + stage + " output");
        }
        return values;
    }

    static final class Pin implements AutoCloseable {
        private final Tensor[] tensors;
        private Pin(Tensor[] tensors) {
            this.tensors = tensors;
            for (Tensor tensor : tensors) { tensor.retainDeviceBuffer(); }
        }
        @Override public void close() {
            for (Tensor tensor : tensors) { tensor.releaseDeviceBufferRetention(); }
        }
    }

    static Tensor finish(Tensor output) {
        if (!GradMode.isEnabled()) {
            output.materialize();
            output.retainDeviceBuffer();
            try { Tensor.backend().releaseTemporaryResources(); }
            finally { output.releaseDeviceBufferRetention(); }
        }
        return output;
    }
}
