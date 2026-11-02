package io.github.kirstenali.deepj.distributed;

/** A failed collective invalidates the group; callers must end this training run. */
public final class CollectiveException extends RuntimeException {
    public CollectiveException(String message) { super(message); }
    public CollectiveException(String message, Throwable cause) { super(message, cause); }
}
