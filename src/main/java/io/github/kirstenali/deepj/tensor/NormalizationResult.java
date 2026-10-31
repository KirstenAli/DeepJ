package io.github.kirstenali.deepj.tensor;

/** The output uses promoted storage; normalized values and denominators retain FLOAT32 accumulation. */
public record NormalizationResult(Tensor output, Tensor normalized, Tensor denominator) {}
