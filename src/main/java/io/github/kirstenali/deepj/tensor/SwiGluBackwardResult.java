package io.github.kirstenali.deepj.tensor;

public record SwiGluBackwardResult(Tensor gateGradient, Tensor upGradient) {}
