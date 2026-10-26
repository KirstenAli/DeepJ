package io.github.kirstenali.deepj.tensor;

/** Integer metadata, typed gather and ordered per-destination FLOAT32 scatter accumulation. */
abstract class GraphIndexRecordingOps extends GraphSliceRecordingOps {
    GraphIndexRecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor layoutIndexSelect(Tensor input, int axis, int[] indices) {
        TensorStorage.requireSource(input);
        var selection = new IndexSelection(input.layout().shape(), axis, indices);
        if (selection.output().elements() == 0) {
            return Tensor.deviceOnly(input.dtype(), selection.output().dimensions());
        }
        requireLayoutSupport();
        return indexSelectCommand(input, indices, selection);
    }

    private Tensor indexSelectCommand(Tensor input, int[] indices, IndexSelection selection) {
        GpuBuffer source = ensureStorageBuffer(input);
        GpuBuffer output = layoutOutput(input.dtype(), selection.output().dimensions());
        int layout = layouts.add(input), plan = layouts.addIndices(indices);
        beginOp(6);
        emitInt(OP_LAYOUT_INDEX_SELECT);
        emitStorageBuffer(source);
        emitStorageBuffer(output);
        emitInt(layout);
        emitInt(plan);
        emitInt(selection.axis());
        endOp();
        return createOutputTensor(output, selection.output().dimensions());
    }

    public void layoutIndexAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        TensorStorage.requireSource(target);
        TensorStorage.requireSource(updates);
        var selection = IndexSelection.requireUpdates(target, updates, axis, indices);
        if (updates.numel() == 0) return;
        requireLayoutSupport();
        Tensor source = target.memory() == updates.memory() ? copyToDevice(updates) : updates;
        indexAddCommand(target, source, selection.axis(), indices);
    }

    private void indexAddCommand(Tensor target, Tensor updates, int axis, int[] indices) {
        GpuBuffer output = ensureStorageBuffer(target), input = ensureStorageBuffer(updates);
        int destination = layouts.add(target), source = layouts.add(updates), plan = layouts.addIndices(indices);
        beginOp(7);
        emitInt(OP_LAYOUT_INDEX_ADD);
        emitStorageBuffer(output);
        emitStorageBuffer(input);
        emitInt(destination);
        emitInt(source);
        emitInt(plan);
        emitInt(axis);
        endOp();
        markUpdated(target);
    }

    public Tensor layoutIndexGradient(Tensor gradient, int[] shape, int axis, int[] indices) {
        var selection = new IndexSelection(new TensorShape(shape), axis, indices);
        if (!selection.output().equals(gradient.layout().shape())) {
            throw new IllegalArgumentException("Indexed gradient must match selected shape");
        }
        if (gradient.numel() != 0) {
            requireLayoutSupport();
            ensureStorageBuffer(gradient);
        }
        Tensor result = zeroToDevice(DType.FLOAT32, shape);
        layoutIndexAdd(result, axis, indices, gradient);
        return result;
    }
}
