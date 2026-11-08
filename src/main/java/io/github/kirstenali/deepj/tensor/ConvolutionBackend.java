package io.github.kirstenali.deepj.tensor;

/** Optional same-padded NCHW cross-correlation with OIHW weights, without a backward graph. */
public interface ConvolutionBackend extends TensorBackend {
    boolean supportsConvolution2D(Tensor input, Tensor weight, int groups);

    Tensor convolution2D(Tensor input, Tensor weight, int groups);
}
