package io.github.kirstenali.deepj.distributed;

/** Per-worker data, random, or recurrent state. Validation must not mutate the running task. */
public interface WorkerState {
    byte[] snapshot();
    void validate(byte[] bytes);
    void restore(byte[] bytes);

    static WorkerState empty() {
        return new WorkerState() {
            @Override public byte[] snapshot() { return new byte[0]; }
            @Override public void validate(byte[] bytes) {
                if (bytes.length != 0) throw new IllegalArgumentException("Unexpected worker state");
            }
            @Override public void restore(byte[] bytes) { validate(bytes); }
        };
    }
}
