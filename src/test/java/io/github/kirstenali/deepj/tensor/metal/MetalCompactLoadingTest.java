package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCompactLoadingTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void loadedCompactProjectionsPreserveLogicalValuesTrainingAndArchives(DType type) throws Exception {
        Tensor slot = metal.zerosForLoading(type, 1024, 256);
        assertFalse(slot.isContiguous());
        Tensor source = Tensor.random(type, new java.util.Random(3), 1024, 256);
        TensorEncoding.restore(slot, source);
        assertArrayEquals(source.toFloatArray(), slot.toFloatArray());
        var actual = new Parameter("weight", slot);
        var expected = new Parameter("weight", metal.copy(source));
        update(actual);
        update(expected);
        assertArrayEquals(expected.value.toFloatArray(), actual.value.toFloatArray());
        assertEquals(DType.FLOAT32, actual.grad().dtype());
        roundTrip(actual, type);
    }

    private void roundTrip(Parameter actual, DType type) throws Exception {
        Path file = directory.resolve("weights.dj");
        ModelSerializer.save(List.of(actual), file);
        var restored = new Parameter("weight", metal.zerosForLoading(type, 1024, 256));
        ModelSerializer.load(List.of(restored), file);
        assertArrayEquals(actual.value.toFloatArray(), restored.value.toFloatArray());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void ordinaryZerosAndLookupTablesRetainContiguousStorage(DType type) {
        assertTrue(metal.zeros(type, 1024, 256).isContiguous());
        assertTrue(metal.zerosForLoading(type, 4096, 64).isContiguous());
        assertTrue(metal.zerosForLoading(type, 2, 32, 32).isContiguous());
        assertTrue(metal.zerosForLoading(type, 1, 512).isContiguous());
        if (type == DType.FLOAT32) assertTrue(metal.zerosForLoading(type, 1024, 256).isContiguous());
    }

    private static void update(Parameter parameter) {
        parameter.value.sumTensor().multiplyScalar(.5f).backward();
        AdamW.defaultAdamW(.001f).step(List.of(parameter));
    }
}
