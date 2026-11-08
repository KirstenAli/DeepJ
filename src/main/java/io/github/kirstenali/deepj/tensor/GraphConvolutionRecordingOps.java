package io.github.kirstenali.deepj.tensor;

/** Native convolution keeps typed buffers in the ordinary deferred command stream. */
abstract class GraphConvolutionRecordingOps extends GraphLayoutRecordingOps {
    GraphConvolutionRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor convolution2D(Tensor input, Tensor weight, int groups) {
        ConvolutionChecks.requireSamePadding(input, weight, groups);
        requireLayoutSupport();
        Tensor source = canonical(input), kernel = canonical(weight);
        int[] shape = {input.size(0), weight.size(0), input.size(2), input.size(3)};
        GpuBuffer output = layoutOutput(input.dtype(), shape);
        recordConvolution(source, kernel, output, groups);
        return createOutputTensor(output, shape);
    }

    private Tensor canonical(Tensor input) {
        return input.coversStorage() ? input : ((GraphExecution) this).copyToDevice(input);
    }

    private void recordConvolution(Tensor input, Tensor weight, GpuBuffer output, int groups) {
        GpuBuffer source = ensureStorageBuffer(input), kernel = ensureStorageBuffer(weight);
        beginOp(13);
        emitInt(OP_CONVOLUTION_2D);
        emitStorageBuffer(source);
        emitStorageBuffer(kernel);
        emitStorageBuffer(output);
        for (int size : input.shape()) {
            emitInt(size);
        }
        emitInt(weight.size(0));
        emitInt(weight.size(2));
        emitInt(weight.size(3));
        emitInt(groups);
        emitInt(input.dtype().storageCode());
        endOp();
    }
}
