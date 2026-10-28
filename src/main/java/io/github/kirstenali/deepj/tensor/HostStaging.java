package io.github.kirstenali.deepj.tensor;

/** Exclusive host payload handoff for private checkpoint staging, never ordinary caller tensors. */
final class HostStaging {
    private static final ThreadLocal<Tensor> SOURCE = new ThreadLocal<>();

    private HostStaging() {}

    static Tensor copy(Tensor source) {
        Tensor previous = SOURCE.get();
        SOURCE.set(source);
        try { return Tensor.backend().copy(source); }
        finally {
            if (previous == null) SOURCE.remove();
            else SOURCE.set(previous);
        }
    }

    static boolean owns(Tensor source) { return SOURCE.get() == source && source.coversStorage(); }
}
