package io.github.kirstenali.deepj.tensor;

/** Test-only access to lazy allocation without fabricating an invalid backing-array length. */
public final class TensorTestStorage {
    private TensorTestStorage() {}

    public static Tensor lazy(int rows, int columns) { return Tensor.deviceOnly(rows, columns); }
    public static boolean hasHostStorage(Tensor tensor) { return tensor.memory().hasHostStorage(); }

    public static Tensor from16Bits(short[] bits, DType dtype, int... shape) {
        Tensor result = new Tensor(dtype, shape);
        if (result.numel() != bits.length) throw new IllegalArgumentException("Raw bits must match shape");
        System.arraycopy(bits, 0, result.memory().shorts(), 0, bits.length);
        return result;
    }

    public static short[] bits16(Tensor tensor) {
        tensor.materialize();
        if (!tensor.coversStorage()) throw new IllegalArgumentException("Raw test access requires complete storage");
        return tensor.memory().shorts().clone();
    }
}
