package io.github.kirstenali.deepj.onnx;

/** Import and control-flow resource bounds, independent of the selected tensor backend. */
public record OnnxOptions(int maxModelBytes, long maxWeightBytes, int maxLoopIterations) {
    public OnnxOptions {
        if (maxModelBytes < 1 || maxWeightBytes < 1 || maxLoopIterations < 1) {
            throw new IllegalArgumentException("ONNX resource limits must be positive");
        }
    }

    public static OnnxOptions defaults() {
        return new OnnxOptions(256 << 20, 2L << 30, 10_000);
    }
}
