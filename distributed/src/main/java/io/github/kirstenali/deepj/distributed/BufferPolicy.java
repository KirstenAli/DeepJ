package io.github.kirstenali.deepj.distributed;

/** BatchNorm training statistics remain local; these policies control persistent buffers between steps. */
public enum BufferPolicy {
    LOCAL,
    BROADCAST_FROM_ROOT
}
