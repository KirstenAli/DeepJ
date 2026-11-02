package io.github.kirstenali.deepj.distributed;

/** Returns up to the requested number of local samples; null marks an exhausted shard. */
@FunctionalInterface
public interface TensorBatchSource {
    TensorBatch nextBatch(int batchSize);
}
