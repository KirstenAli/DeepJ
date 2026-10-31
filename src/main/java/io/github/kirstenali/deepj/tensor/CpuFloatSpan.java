package io.github.kirstenali.deepj.tensor;

/** Internal contiguous FLOAT32 access; logical tensor shapes and alias offsets stay unchanged. */
record CpuFloatSpan(float[] values, int offset) {
    static CpuFloatSpan of(Tensor tensor) {
        return new CpuFloatSpan(tensor.memory().floats(), tensor.storageOffset());
    }

    float get(int index) { return values[offset + index]; }
    void set(int index, float value) { values[offset + index] = value; }
}
