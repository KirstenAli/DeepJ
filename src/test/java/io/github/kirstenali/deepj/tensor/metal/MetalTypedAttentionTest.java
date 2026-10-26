package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedAttentionTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void groupedBatchAttentionUsesTypedStorageAndFloat32GradientsWithoutHostNumericalFallback(DType dtype) {
        Tensor.setBackend(cpu);
        Tensor q = values(dtype, 2, 4, 4, 2), k = values(dtype, 2, 2, 4, 2), v = k.detach().addScalar(0.125f);
        Result expected = run(q, k, v);
        Tensor.setBackend(metal);
        Tensor query = metal.copy(q), key = metal.copy(k), value = metal.copy(v);
        Result actual = run(query, key, value);
        assertFalse(TensorTestStorage.hasHostStorage(query));
        assertFalse(TensorTestStorage.hasHostStorage(key));
        compare(expected, actual, dtype);
    }

    private static void compare(Result expected, Result actual, DType dtype) {
        assertEquals(dtype, actual.output().dtype());
        assertTensorClose(expected.output(), actual.output(), 3e-6f, 0);
        assertGradientListsClose(expected.gradients(), actual.gradients(), 3e-5f, 5e-4f);
        for (Tensor gradient : actual.gradients()) { assertEquals(DType.FLOAT32, gradient.dtype()); }
    }

    private static Result run(Tensor query, Tensor key, Tensor value) {
        query.requiresGrad(true);
        key.requiresGrad(true);
        value.requiresGrad(true);
        Tensor output = AttentionOps.causal(query, key, value, 4, 2, 0.5f);
        output.sumTensor().backward();
        return new Result(output, List.of(query.grad().detach(), key.grad().detach(), value.grad().detach()));
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = 1;
        for (int size : shape) { count *= size; }
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = (index % 13 - 6) * 0.03125f; }
        return Tensor.from(values, dtype, shape);
    }

    private record Result(Tensor output, List<Tensor> gradients) {}
}
