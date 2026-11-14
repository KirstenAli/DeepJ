package io.github.kirstenali.deepj.tensor;

abstract class GraphQ8RecordingOps extends GraphExecution {
    GraphQ8RecordingOps(GpuRuntime runtime) { super(runtime); }

    public Tensor projectQ8(Tensor input, Tensor packed, int rows, int width) {
        requireQ8(packed, rows, width);
        if (input.rank() != 2 || input.size(1) != width) throw new IllegalArgumentException("Invalid Q8_0 input");
        if (input.size(0) == 0) return zeroToDevice(DType.FLOAT32, 0, rows);
        try (var scope = TensorScope.open()) {
            Tensor source = input.coversStorage() ? input : copyToDevice(input);
            source = castToDevice(source, DType.FLOAT32);
            return scope.escape(recordQ8(OP_Q8_PROJECT, source, packed, rows, width, input.size(0), rows));
        }
    }

    public Tensor gatherQ8(Tensor packed, int rows, int width, int[] tokens) {
        requireQ8(packed, rows, width);
        if (tokens.length == 0) return zeroToDevice(DType.FLOAT32, 0, width);
        try (var scope = TensorScope.open()) {
            float[] indices = new float[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                if (tokens[i] < 0 || tokens[i] >= rows || tokens[i] >= (1 << 24)) {
                    throw new IllegalArgumentException("Invalid Q8_0 embedding row");
                }
                indices[i] = tokens[i];
            }
            Tensor input = Tensor.from(indices, 1, indices.length);
            return scope.escape(recordQ8(OP_Q8_GATHER, input, packed, rows, width, tokens.length, width));
        }
    }

    private Tensor recordQ8(int opcode, Tensor input, Tensor packed, int rows, int width, int batch, int columns) {
        GpuBuffer source = ensureStorageBuffer(input), weights = ensureStorageBuffer(packed);
        GpuBuffer output = newOutputBuffer(batch, columns);
        beginOp(7);
        emitInt(opcode);
        emitStorageBuffer(source);
        emitStorageBuffer(weights);
        emitStorageBuffer(output);
        emitInt(rows);
        emitInt(width);
        emitInt(batch);
        endOp();
        return createOutputTensor(output);
    }

    private static void requireQ8(Tensor packed, int rows, int width) {
        if (rows <= 0 || width <= 0 || width % 32 != 0 || packed.dtype() != DType.FLOAT16
                || !packed.coversStorage() || packed.numel() != (long) rows * (width / 32) * 17) {
            throw new IllegalArgumentException("Invalid packed Q8_0 buffer");
        }
    }
}
