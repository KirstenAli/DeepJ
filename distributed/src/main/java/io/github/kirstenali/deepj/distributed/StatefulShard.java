package io.github.kirstenali.deepj.distributed;

import java.util.Objects;

/** Shared cursor state for tensor and token data sources. */
public abstract class StatefulShard implements WorkerState {
    protected final ShardSampler sampler;
    protected StatefulShard(ShardSampler sampler) { this.sampler = Objects.requireNonNull(sampler, "sampler"); }
    public final ShardSampler sampler() { return sampler; }
    @Override public final byte[] snapshot() { return sampler.snapshot(); }
    @Override public final void validate(byte[] bytes) { sampler.validate(bytes); }
    @Override public final void restore(byte[] bytes) { sampler.restore(bytes); }
}
