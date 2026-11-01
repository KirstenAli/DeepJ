package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Identifies library-owned inference separately from arbitrary caller callbacks. */
record UncachedGenerationSession(DecoderOnlyModel model) implements GenerationSession {
    @Override public Tensor apply(int[] tokens) { return model.forward(tokens); }
}
