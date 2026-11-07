package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class InferenceContextGrowthTest {
    private final TensorBackend previous = Tensor.backend();
    private MetalBackend metal;

    @AfterEach void restore() {
        if (metal != null) metal.releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void growingPreservesBatchesGroupedHeadsAndAllCachedValuesAfterClosingOldStorage(String backend) {
        if (backend.equals("metal")) {
            assumeTrue(MetalBackend.isAvailable(), "Metal device required");
            metal = new MetalBackend();
        }
        Tensor.setBackend(metal == null ? new CpuBackend() : metal);
        for (DType dtype : DType.values()) { compare(dtype); }
    }

    private static void compare(DType dtype) {
        var growing = new InferenceContext(3);
        try (var reference = new InferenceContext(16); var ignored = GradMode.noGrad()) {
            Object owner = new Object();
            for (int offset = 0; offset < 12; offset++) {
                if (offset == 3 || offset == 7) growing = grow(growing, offset == 3 ? 7 : 16);
                Tensor query = Tensor.random(dtype, new Random(19 + offset), 2, 4, 1, 8);
                Tensor key = Tensor.random(dtype, new Random(39 + offset), 2, 2, 1, 8);
                Tensor value = Tensor.random(dtype, new Random(59 + offset), 2, 2, 1, 8);
                assertArrayEquals(attend(reference, owner, offset, query, key, value),
                        attend(growing, owner, offset, query, key, value));
                Tensor.backend().releaseTemporaryResources();
            }
            InferenceContext active = growing;
            assertThrows(IllegalArgumentException.class, () -> active.grown(16));
        } finally { growing.close(); }
    }

    private static InferenceContext grow(InferenceContext source, int capacity) {
        InferenceContext result = source.grown(capacity);
        source.close();
        return result;
    }

    private static float[] attend(InferenceContext context, Object owner, int offset,
                                   Tensor query, Tensor key, Tensor value) {
        context.begin(offset);
        Tensor output = context.attend(owner, query, key, value, 4, 2, .125f);
        context.finish();
        return output.toFloatArray();
    }

    @Test void wrappedCachesRejectGrowthAndRemainUsable() {
        Tensor.setBackend(new CpuBackend());
        Tensor row = Tensor.ones(DType.FLOAT16, 1, 1, 1, 2);
        Object owner = new Object();
        try (var cache = new InferenceContext(2, true); var ignored = GradMode.noGrad()) {
            for (int offset = 0; offset < 3; offset++) {
                cache.begin(offset);
                cache.attend(owner, row, row, row, 1, 1, 1).materialize();
                cache.finish();
            }
            assertThrows(IllegalArgumentException.class, () -> cache.grown(4));
            cache.begin(3);
            assertArrayEquals(new float[]{1, 1}, cache.attend(owner, row, row, row, 1, 1, 1).toFloatArray());
            cache.finish();
        }
    }
}
