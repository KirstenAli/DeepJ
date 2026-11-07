package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ConvolutionBackend;
import io.github.kirstenali.deepj.tensor.ConvolutionChecks;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

public final class MetalBackend extends MetalAttentionOps implements ConvolutionBackend {
    @Override public boolean supportsConvolution2D(Tensor input, Tensor weight, int groups) {
        return ConvolutionChecks.samePadding(input, weight, groups)
                && (input.dtype() != DType.BFLOAT16 || ConvolutionCapabilities.BFLOAT);
    }

    @Override public Tensor convolution2D(Tensor input, Tensor weight, int groups) {
        if (!supportsConvolution2D(input, weight, groups)) {
            throw new IllegalArgumentException("Expected matching NCHW input and odd OIHW kernels");
        }
        return graph.convolution2D(input, weight, groups);
    }

    private static final class ConvolutionCapabilities {
        private static final boolean BFLOAT = MetalNative.nativeSupportsCompactGraph();
        private ConvolutionCapabilities() {}
    }
}
