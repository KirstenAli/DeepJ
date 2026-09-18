package io.github.kirstenali.deepj.data;

/** Supplies token batches to a causal language-model trainer. */
@FunctionalInterface
public interface BatchSource {

    Batch nextBatch(int batchSize);
}
