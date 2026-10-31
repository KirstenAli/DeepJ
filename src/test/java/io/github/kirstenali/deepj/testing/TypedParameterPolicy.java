package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.DType;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Mixed-precision policy: matrices take the requested dtype, 1xN vectors and activations stay FLOAT32. */
public final class TypedParameterPolicy {
    private TypedParameterPolicy() {}

    public static DType expected(DType requested, Parameter parameter) {
        return parameter.value.size(0) == 1 ? DType.FLOAT32 : requested;
    }

    public static void assertStorage(DType requested, Parameter parameter) {
        DType expected = expected(requested, parameter);
        assertEquals(expected, parameter.value.dtype(), parameter.name());
        assertEquals(expected.storageBytes(parameter.value.numel()), parameter.value.storageBytes(), parameter.name());
        assertEquals(DType.FLOAT32, parameter.grad().dtype(), parameter.name());
    }
}

