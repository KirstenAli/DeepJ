package io.github.kirstenali.deepj.tensor;

/** General GPU arithmetic uses logical layouts and retains genuine typed allocations. */
abstract class GraphElementwiseRecordingOps extends GraphLayoutRecordingOps {
    GraphElementwiseRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutBinary(Tensor left, Tensor right, int operation) {
        requireBinary(operation);
        TensorStorage.requireSource(left);
        TensorStorage.requireSource(right);
        int[] shape = ShapeTransforms.broadcast(left.layout().shape(), right.layout().shape()).dimensions();
        DType dtype = DType.promote(left.dtype(), right.dtype());
        if (new TensorShape(shape).elements() == 0) return Tensor.deviceOnly(dtype, shape);
        requireLayoutSupport();
        return binaryCommand(TensorViews.expand(left, shape), TensorViews.expand(right, shape), operation);
    }

    private Tensor binaryCommand(Tensor left, Tensor right, int operation) {
        GpuBuffer a = ensureStorageBuffer(left), b = ensureStorageBuffer(right);
        GpuBuffer output = layoutOutput(DType.promote(left.dtype(), right.dtype()), left.shape());
        int first = layouts.add(left), second = layouts.add(right);
        beginOp(7);
        emitInt(OP_LAYOUT_BINARY);
        emitStorageBuffer(a);
        emitStorageBuffer(b);
        emitStorageBuffer(output);
        emitInt(first);
        emitInt(second);
        emitInt(operation);
        endOp();
        return createOutputTensor(output, left.shape());
    }

    public Tensor layoutUnary(Tensor input, int operation, float first, float second) {
        requireUnary(operation);
        TensorStorage.requireSource(input);
        if (input.numel() == 0) return Tensor.deviceOnly(input.dtype(), input.shape());
        requireLayoutSupport();
        return unaryCommand(input, operation, first, second);
    }

    private Tensor unaryCommand(Tensor input, int operation, float first, float second) {
        GpuBuffer source = ensureStorageBuffer(input), output = layoutOutput(input.dtype(), input.shape());
        int layout = layouts.add(input);
        beginOp(7);
        emitInt(OP_LAYOUT_UNARY);
        emitStorageBuffer(source);
        emitStorageBuffer(output);
        emitInt(layout);
        emitInt(operation);
        emitFloatBits(first);
        emitFloatBits(second);
        endOp();
        return createOutputTensor(output, input.shape());
    }

    public void assignToDevice(Tensor target, Tensor source) {
        TensorStorage.requireSource(target);
        TensorStorage.requireSource(source);
        Tensor.requireSameShape(target, source, "GPU assignment");
        target.layout().requireWritable();
        if (target.numel() == 0) return;
        requireLayoutSupport();
        Tensor snapshot = target.memory() == source.memory() ? copyToDevice(source) : source;
        assignCommand(target, snapshot);
    }

    public void layoutBinaryInto(Tensor target, Tensor operand, int operation) {
        TensorStorage.requireSource(target);
        TensorStorage.requireSource(operand);
        target.layout().requireWritable();
        TensorShape shape = ShapeTransforms.broadcast(target.layout().shape(), operand.layout().shape());
        if (!shape.equals(target.layout().shape())) throw new IllegalArgumentException("Cannot expand in-place output");
        if (DType.promote(target.dtype(), operand.dtype()) != target.dtype()) {
            throw new IllegalArgumentException("In-place output cannot narrow the promoted dtype");
        }
        assignToDevice(target, layoutBinary(target, operand, operation));
    }

    public void layoutUnaryInto(Tensor target, int operation, float scalar) {
        TensorStorage.requireSource(target);
        target.layout().requireWritable();
        assignToDevice(target, layoutUnary(target, operation, scalar, 0));
    }

    private void assignCommand(Tensor target, Tensor source) {
        GpuBuffer input = ensureStorageBuffer(source), output = ensureStorageBuffer(target);
        int first = layouts.add(source), second = layouts.add(target);
        beginOp(5);
        emitInt(OP_LAYOUT_ASSIGN);
        emitStorageBuffer(input);
        emitStorageBuffer(output);
        emitInt(first);
        emitInt(second);
        endOp();
        markUpdated(target);
    }

    private static void requireBinary(int operation) {
        switch (operation) {
            case OP_ADD, OP_SUBTRACT, OP_MULTIPLY, OP_DIVIDE, OP_RELU_BACKWARD, OP_GELU_BACKWARD -> { }
            default -> throw new IllegalArgumentException("Unsupported GPU binary operation: " + operation);
        }
    }

    private static void requireUnary(int operation) {
        switch (operation) {
            case OP_MULTIPLY_SCALAR, OP_ADD_SCALAR, OP_DIVIDE_SCALAR, OP_SQRT, OP_NEG, OP_EXP, OP_LOG,
                    OP_TANH, OP_SIGMOID, OP_RELU, OP_GELU, OP_CLAMP, OP_POW -> { }
            default -> throw new IllegalArgumentException("Unsupported GPU unary operation: " + operation);
        }
    }
}
