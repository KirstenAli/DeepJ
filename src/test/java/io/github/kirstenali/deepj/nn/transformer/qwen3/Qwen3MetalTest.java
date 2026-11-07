package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3MetalTest {
    private MetalBackend metal;
    @TempDir Path directory;

    @BeforeEach void device() {
        boolean available = MetalBackend.isAvailable();
        if (Boolean.getBoolean("deepj.qwen3.requireMetal")) {
            assertTrue(available, "A physical Metal device is required");
        }
        Assumptions.assumeTrue(available, "Metal device not available");
        metal = new MetalBackend();
        Tensor.setBackend(metal);
    }

    @AfterEach void cleanup() {
        if (metal != null) metal.releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactActivationsMatchOfficialTransformersOnMetal(DType dtype) throws IOException {
        Qwen3CompactTest.compareReference(dtype);
    }

    @ParameterizedTest @ValueSource(strings = {"float32", "float16", "bfloat16"})
    void officialReferenceLogitsGradientsAndCachedDecodingRunOnMetal(String dtype) throws IOException {
        Qwen3 model = Qwen3Pretrained.load(Qwen3Fixtures.path(dtype));
        assertFalse(model.parameters().stream().anyMatch(parameter ->
                parameter.value.matrixRows() > 1 && parameter.value.hasHostStorage()),
                "Loading weight matrices on Metal should not retain host arrays");
        var expected = Qwen3Fixtures.expected(dtype);
        Tensor logits = model.forwardBatch(Qwen3Fixtures.tokens(expected));
        assertNotNull(logits.getGpuTag(), "Logits must be computed on Metal");
        assertArrayEquals(Qwen3Fixtures.floats(expected.get("logits")), logits.toFloatArray(), 2e-5f);
        logits.backward(Tensor.from(Qwen3Fixtures.floats(expected.get("upstream")), 2, 4, 19));
        compareGradients(model, expected);
        cached(model, Qwen3Fixtures.tokens(expected)[0]);
    }

    private static void compareGradients(Qwen3 model, java.util.Map<String, Object> expected) {
        var weights = Qwen3Weights.inventory(model.config()).stream()
                .collect(java.util.stream.Collectors.toMap(Qwen3Weights.Weight::name, weight -> weight));
        for (var parameter : model.parameters()) {
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            assertArrayEquals(Qwen3Fixtures.gradient(weights.get(parameter.name()), expected),
                    parameter.grad().toFloatArray(), 0.0001f, parameter.name());
        }
    }

    private static void cached(Qwen3 model, int[] tokens) {
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 1; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                assertArrayEquals(model.forward(prefix).select(0, length - 1).toFloatArray(),
                        session.apply(prefix).toFloatArray(), 2e-5f);
            }
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void trainingAndCheckpointLoadingKeepOtherLiveModelsValid(DType dtype) throws IOException {
        Qwen3 model = new Qwen3(Qwen3ModelTest.tiny(), 3, dtype);
        Batch batch = new Batch(new int[][]{{1, 2, 3}, {4, 5, 6}}, new int[][]{{2, 3, 4}, {5, 6, 7}});
        var trainer = CausalLMTraining.trainer(model, size -> batch, AdamW.defaultAdamW(0.005f));
        float first = trainer.trainStep(2), last = first;
        for (int step = 0; step < 5; step++) { last = trainer.trainStep(2); }
        assertTrue(last < first);
        try (var ignored = GradMode.noGrad()) {
            float[] before = model.forward(new int[]{1, 2, 3}).toFloatArray();
            Qwen3Checkpoint.save(model, directory.resolve("weights"));
            Qwen3 loaded = Qwen3Checkpoint.load(directory.resolve("weights"));
            assertArrayEquals(before, loaded.forward(new int[]{1, 2, 3}).toFloatArray(), 2e-5f);
            assertArrayEquals(before, model.forward(new int[]{1, 2, 3}).toFloatArray());
        }
    }
}
