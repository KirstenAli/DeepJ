package io.github.kirstenali.deepj.tensor;

/** Forward output and row maximum/denominator pairs saved for tiled backward. */
public record CausalAttentionResult(Tensor output, Tensor normalization) {}
