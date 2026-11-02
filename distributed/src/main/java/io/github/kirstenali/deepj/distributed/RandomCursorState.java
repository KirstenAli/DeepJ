package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.StatefulBatchSource;
import java.nio.ByteBuffer;

final class RandomCursorState implements WorkerState {
    private final StatefulBatchSource source;
    RandomCursorState(StatefulBatchSource source) { this.source = source; }
    @Override public byte[] snapshot() { return ByteBuffer.allocate(Long.BYTES).putLong(source.randomState()).array(); }
    @Override public void validate(byte[] bytes) {
        if (bytes.length != Long.BYTES) throw new IllegalArgumentException("Invalid dataset random state");
    }
    @Override public void restore(byte[] bytes) {
        validate(bytes);
        source.restoreRandomState(ByteBuffer.wrap(bytes).getLong());
    }
}
