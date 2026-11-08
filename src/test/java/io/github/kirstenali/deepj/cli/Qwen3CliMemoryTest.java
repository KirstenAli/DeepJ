package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.requireMetal", matches = "true")
class Qwen3CliMemoryTest {
    @TempDir Path directory;

    @Test void closingDiscardsOwnedWeightsWithoutMaterializingThemOrLosingOtherTensorValues() throws Exception {
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("model"), DType.BFLOAT16);
        var loaded = (Qwen3CliModel) CliModel.load(snapshot, BackendChoice.METAL, DType.BFLOAT16);
        var weights = loaded.chat().model().parameters().stream().map(parameter -> parameter.value)
                .filter(value -> !value.hasHostStorage()).toList();
        Tensor other = Tensor.from(new float[]{2, 3}).multiplyScalar(2);
        try {
            assertFalse(weights.isEmpty());
            loaded.close();
            assertTrue(weights.stream().noneMatch(Tensor::hasHostStorage));
            assertTrue(weights.stream().allMatch(value -> value.getGpuTag() == null));
            assertArrayEquals(new float[]{4, 6}, other.toFloatArray());
            assertDoesNotThrow(loaded::close);
        } finally { Tensor.backend().releaseResources(); }
    }
}
