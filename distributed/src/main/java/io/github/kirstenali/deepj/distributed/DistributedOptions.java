package io.github.kirstenali.deepj.distributed;

import java.util.Objects;

public record DistributedOptions(int bucketElements, BufferPolicy bufferPolicy) {
    public DistributedOptions {
        if (bucketElements < 1 || bucketElements > Wire.MAX_PAYLOAD / Float.BYTES) {
            throw new IllegalArgumentException("Bucket must fit one collective frame");
        }
        Objects.requireNonNull(bufferPolicy, "bufferPolicy");
    }

    public static DistributedOptions defaults() {
        return new DistributedOptions(262144, BufferPolicy.BROADCAST_FROM_ROOT);
    }
}
