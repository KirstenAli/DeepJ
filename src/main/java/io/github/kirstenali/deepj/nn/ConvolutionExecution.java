package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.ConvolutionBackend;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Keeps differentiable convolution composition separate from optional inference kernels. */
final class ConvolutionExecution {
    private ConvolutionExecution() {}

    static Tensor inference(Tensor input, Tensor weight, int groups) {
        if (GradMode.isEnabled()) return null;
        if (!(Tensor.backend() instanceof ConvolutionBackend backend)) return null;
        if (!backend.supportsConvolution2D(input, weight, groups)) return null;
        ModelMemory.checkCancelled();
        if (ModelInference.active()) return backend.convolution2D(input, weight, groups);
        try (var buffers = BackwardBuffers.begin()) {
            Tensor result = buffers.capture(() -> backend.convolution2D(input, weight, groups));
            buffers.expose(result);
            buffers.releaseTemporaryResources();
            return result;
        }
    }
}
