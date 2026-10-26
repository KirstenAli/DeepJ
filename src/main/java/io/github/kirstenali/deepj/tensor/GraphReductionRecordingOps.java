package io.github.kirstenali.deepj.tensor;

/** Grouped-axis GPU reductions preserve logical coordinate order and FLOAT32 accumulation. */
abstract class GraphReductionRecordingOps extends GraphElementwiseRecordingOps {
    GraphReductionRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutReduction(Tensor input, int[] axes, boolean keep, int operation, int correction) {
        TensorStorage.requireSource(input);
        return reduce(input, new ReductionShape(input.layout().shape(), axes, keep),
                input.dtype(), operation, correction);
    }

    public Tensor layoutTotal(Tensor input, int operation) {
        TensorStorage.requireSource(input);
        if (operation != 0 && operation != 4 && operation != 5) {
            throw new IllegalArgumentException("Invalid scalar accumulation operation");
        }
        return reduce(input, new ReductionShape(input.layout().shape(), new int[0], false),
                DType.FLOAT32, operation, 0);
    }

    private Tensor reduce(Tensor input, ReductionShape shape, DType dtype, int operation, int correction) {
        requireReduction(operation, correction);
        if (operation == 3 && shape.count() == 0) throw new IllegalArgumentException("Empty maximum reduction");
        int[] output = shape.output().dimensions();
        if (shape.output().elements() == 0) return Tensor.deviceOnly(dtype, output);
        if (shape.count() == 0 || operation == 2 && shape.count() <= correction) {
            return emptyReduction(dtype, output, operation);
        }
        requireLayoutSupport();
        Tensor grouped = TensorViews.permute(input, shape.groupedAxes());
        return reduceCommand(grouped, shape, dtype, operation, correction);
    }

    private Tensor emptyReduction(DType dtype, int[] shape, int operation) {
        return operation == 0 || operation >= 4 ? zeroToDevice(dtype, shape) : nanToDevice(dtype, shape);
    }

    private Tensor reduceCommand(Tensor input, ReductionShape shape, DType dtype, int operation, int correction) {
        GpuBuffer source = ensureStorageBuffer(input), output = layoutOutput(dtype, shape.output().dimensions());
        int layout = layouts.add(input);
        beginOp(7);
        emitInt(OP_LAYOUT_REDUCE);
        emitStorageBuffer(source);
        emitStorageBuffer(output);
        emitInt(layout);
        emitInt(shape.retainedRank());
        emitInt(operation);
        emitInt(correction);
        endOp();
        return createOutputTensor(output, shape.output().dimensions());
    }

    private Tensor nanToDevice(DType dtype, int... shape) {
        // A single constant is staged; no tensor input is downloaded or numerically processed on CPU.
        Tensor scalar = Tensor.from(new float[]{Float.NaN}, dtype, new int[0]);
        return transferLayout(TensorViews.expand(scalar, shape), dtype);
    }

    private static void requireReduction(int operation, int correction) {
        if (operation < 0 || operation > 5) throw new IllegalArgumentException("Invalid GPU reduction operation");
        if (correction < 0 || operation != 2 && correction != 0) {
            throw new IllegalArgumentException("Only variance accepts a nonnegative correction");
        }
    }

    public Tensor layoutSoftmax(Tensor input, int axis) {
        TensorStorage.requireSource(input);
        axis = input.layout().shape().axis(axis);
        if (input.numel() == 0) return Tensor.deviceOnly(input.dtype(), input.shape());
        requireLayoutSupport();
        var shape = new ReductionShape(input.layout().shape(), new int[]{axis}, false);
        Tensor grouped = TensorViews.permute(input, shape.groupedAxes());
        GpuBuffer source = ensureStorageBuffer(grouped);
        Tensor output = createOutputTensor(layoutOutput(input.dtype(), input.shape()), input.shape());
        softmaxCommand(grouped, source, TensorViews.permute(output, shape.groupedAxes()), shape.retainedRank());
        return output;
    }

    private void softmaxCommand(Tensor input, GpuBuffer source, Tensor output, int retained) {
        GpuBuffer target = ensureStorageBuffer(output);
        int first = layouts.add(input), second = layouts.add(output);
        beginOp(6);
        emitInt(OP_LAYOUT_SOFTMAX);
        emitStorageBuffer(source);
        emitStorageBuffer(target);
        emitInt(first);
        emitInt(second);
        emitInt(retained);
        endOp();
    }
}
