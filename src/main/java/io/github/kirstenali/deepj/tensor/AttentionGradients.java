package io.github.kirstenali.deepj.tensor;

public record AttentionGradients(Tensor query, Tensor key, Tensor value) {}
