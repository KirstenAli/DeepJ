package io.github.kirstenali.deepj.tensor;

/** Test-only access to lazy allocation without fabricating an invalid backing-array length. */
public final class TensorTestStorage {
    private TensorTestStorage() {}

    public static Tensor lazy(int rows, int columns) { return Tensor.deviceOnly(rows, columns); }
    public static boolean hasHostStorage(Tensor tensor) { return tensor.memory().hasHostStorage(); }
}
