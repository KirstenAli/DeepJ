package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Batch-first sequence output and the final state of a recurrent layer. */
public record RecurrentResult(Tensor sequence, RecurrentState state) {}
