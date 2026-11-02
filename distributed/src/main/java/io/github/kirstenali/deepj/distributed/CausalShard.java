package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import java.util.Objects;

/** Shards complete token sequences; loss masks retain their sequence association. */
public final class CausalShard extends StatefulShard implements BatchSource {
    private final Batch data;

    public CausalShard(Batch data, ShardSampler sampler) {
        super(sampler);
        this.data = Objects.requireNonNull(data, "data");
        if (data.x() == null || data.y() == null || data.x().length != sampler.samples()
                || data.y().length != sampler.samples()
                || data.lossMask() != null && data.lossMask().length != sampler.samples()) {
            throw new IllegalArgumentException("Shard inputs, targets, and masks require matching sample counts");
        }
    }

    @Override public Batch nextBatch(int batchSize) {
        int[] indices = sampler.next(batchSize);
        if (indices.length == 0) return null;
        int[][] inputs = new int[indices.length][], targets = new int[indices.length][];
        boolean[][] masks = data.lossMask() == null ? null : new boolean[indices.length][];
        for (int row = 0; row < indices.length; row++) {
            inputs[row] = data.x()[indices[row]];
            targets[row] = data.y()[indices[row]];
            if (masks != null) masks[row] = data.lossMask()[indices[row]];
        }
        return new Batch(inputs, targets, masks);
    }
}
