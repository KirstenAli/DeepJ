package io.github.kirstenali.deepj.tensor;

/** Output, row statistics and unrounded FP32 output needed by tiled backward. */
public record CausalAttentionResult(Tensor output, Tensor normalization, Tensor accumulatedOutput) {
    public CausalAttentionResult(Tensor output, Tensor normalization) {
        this(output, normalization, output);
    }
}
