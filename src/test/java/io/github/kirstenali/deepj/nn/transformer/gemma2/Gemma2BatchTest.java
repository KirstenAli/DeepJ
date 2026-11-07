package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2BatchTest {
    private static final Path ROOT = Path.of("src/test/resources/sana/gemma");
    private static final int[][] TOKENS = {{2, 3, 4, 5, 0, 0}, {2, 6, 0, 0, 0, 0}};
    private static final int[][] MASKS = {{1, 1, 1, 1, 0, 0}, {1, 1, 0, 0, 0, 0}};

    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void paddedRowsKeepTheirOwnCausalAndSlidingWindowMasks(DType dtype) throws Exception {
        try (var ignored = GradMode.noGrad(); var weights = TensorWeights.open(ROOT, dtype)) {
            var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("config.json")), weights::tensor);
            float[] first = model.forward(TOKENS[0], MASKS[0]).toFloatArray();
            float[] second = model.forward(TOKENS[1], MASKS[1]).toFloatArray();
            Tensor batch = model.forward(TOKENS, MASKS);
            float tolerance = dtype == DType.FLOAT32 ? 3e-5f : 0.035f;
            assertArrayEquals(first, batch.slice(0, 0, 1).toFloatArray(), tolerance);
            assertArrayEquals(second, batch.slice(0, 1, 2).toFloatArray(), tolerance);
            assertEquals(2, batch.size(0));
            assertEquals(dtype, batch.dtype());
        }
    }

    @Test void batchGradientsAgreeWithTheSumOfIndependentSequenceLosses() throws Exception {
        assertArrayEquals(gradient(false), gradient(true), 1e-4f);
    }

    private static float[] gradient(boolean batched) throws Exception {
        try (var weights = TensorWeights.open(ROOT, DType.FLOAT32)) {
            var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("config.json")), weights::tensor);
            Tensor loss = batched ? model.forward(TOKENS, MASKS).sumTensor()
                    : model.forward(TOKENS[0], MASKS[0]).sumTensor()
                    .add(model.forward(TOKENS[1], MASKS[1]).sumTensor());
            loss.backward();
            return weights.tensor("embed_tokens.weight").grad().toFloatArray();
        }
    }

    @Test void mismatchedOrUnequalRowsAreRejected() throws Exception {
        try (var weights = TensorWeights.open(ROOT, DType.FLOAT32)) {
            var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("config.json")), weights::tensor);
            assertThrows(IllegalArgumentException.class, () -> model.forward(new int[0][], new int[0][]));
            assertThrows(IllegalArgumentException.class, () -> model.forward(TOKENS, new int[][]{MASKS[0]}));
            assertThrows(IllegalArgumentException.class,
                    () -> model.forward(new int[][]{{2}, {2, 3}}, new int[][]{{1}, {1, 1}}));
        }
    }
}
