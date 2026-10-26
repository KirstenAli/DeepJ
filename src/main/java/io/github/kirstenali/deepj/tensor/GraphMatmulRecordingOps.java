package io.github.kirstenali.deepj.tensor;

/** Matmul descriptors promote vectors and broadcast batches without changing physical storage. */
abstract class GraphMatmulRecordingOps extends GraphMaximumRecordingOps {
    GraphMatmulRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutMatmul(Tensor left, Tensor right) {
        TensorStorage.requireSource(left);
        TensorStorage.requireSource(right);
        NdMatmulShape shape = new NdMatmulShape(left, right);
        DType dtype = DType.promote(left.dtype(), right.dtype());
        if (shape.output().elements() == 0) return Tensor.deviceOnly(dtype, shape.output().dimensions());
        if (shape.inner() == 0) return zeroToDevice(dtype, shape.output().dimensions());
        requireLayoutSupport();
        return matmulCommand(Tensor.aliasStorage(left, shape.left()),
                Tensor.aliasStorage(right, shape.right()), shape.output(), dtype);
    }

    private Tensor matmulCommand(Tensor left, Tensor right, TensorShape shape, DType dtype) {
        GpuBuffer first = ensureStorageBuffer(left), second = ensureStorageBuffer(right);
        GpuBuffer output = layoutOutput(dtype, shape.dimensions());
        int a = layouts.add(left), b = layouts.add(right);
        beginOp(6);
        emitInt(OP_LAYOUT_MATMUL);
        emitStorageBuffer(first);
        emitStorageBuffer(second);
        emitStorageBuffer(output);
        emitInt(a);
        emitInt(b);
        endOp();
        return createOutputTensor(output, shape.dimensions());
    }
}
