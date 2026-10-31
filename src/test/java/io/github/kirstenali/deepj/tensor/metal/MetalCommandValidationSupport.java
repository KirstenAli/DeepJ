package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Assertions;
import java.util.ArrayList;
import java.util.List;

/** Retains native input owners while malformed command fixtures are checked independently. */
abstract class MetalCommandValidationSupport extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    protected abstract long[] words();

    protected int buffer(DType dtype, int count) {
        Tensor owner = new Tensor(dtype, count);
        owners.add(owner);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        return id;
    }

    protected static int bits(float value) { return Float.floatToRawIntBits(value); }

    protected void invalidField(int[] original, int field, int value) {
        int[] command = original.clone();
        command[field] = value;
        invalid(command);
    }

    protected void invalid(int[] command) { invalid(command, words()); }

    protected static void invalid(int[] command, long[] words) {
        Assertions.assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(command, command.length, words));
    }
}
