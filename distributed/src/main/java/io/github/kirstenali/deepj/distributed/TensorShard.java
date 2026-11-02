package io.github.kirstenali.deepj.distributed;

import java.util.Objects;

public final class TensorShard extends StatefulShard implements TensorBatchSource {
    private final TensorBatch data;

    public TensorShard(TensorBatch data, ShardSampler sampler) {
        super(sampler);
        this.data = Objects.requireNonNull(data, "data");
        if (data.samples() != sampler.samples()) throw new IllegalArgumentException("Shard data size differs");
    }

    @Override public TensorBatch nextBatch(int batchSize) {
        int[] indices = sampler.next(batchSize);
        if (indices.length == 0) return null;
        return new TensorBatch(data.inputs().indexSelect(0, indices), data.targets().indexSelect(0, indices));
    }
}
