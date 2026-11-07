package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3CompactTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void sharedChatLoaderUsesCompactActivationsOnlyWhenExplicitlyRequested(DType dtype) throws IOException {
        Path snapshot = Qwen3ChatFixture.create(directory.resolve("chat"), dtype);
        var compact = Qwen3ChatModel.load(snapshot, dtype);
        var legacy = Qwen3ChatModel.load(snapshot);
        try (var ignored = GradMode.noGrad()) {
            assertEquals(dtype, compact.model().forward(new int[]{1, 10}).dtype());
            assertEquals(DType.FLOAT32, legacy.model().forward(new int[]{1, 10}).dtype());
        }
        assertEquals("xx", compact.stream("hello", 2, 0.8f, 1, 5, 1,
                io.github.kirstenali.deepj.nn.transformer.ContextWindowMode.REBUILD, text -> {}));
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void weightRoundTripRestoresTheExplicitActivationPolicy(DType dtype) throws IOException {
        var model = new Qwen3(Qwen3ModelTest.tiny(), 3, ParameterFactory.uniform(dtype));
        Path snapshot = directory.resolve("compact");
        Qwen3Checkpoint.save(model, snapshot);
        var restored = Qwen3Pretrained.load(snapshot, null, true);
        try (var ignored = GradMode.noGrad()) {
            Tensor expected = model.forward(new int[]{1, 2, 3});
            Tensor actual = restored.forward(new int[]{1, 2, 3});
            assertEquals(dtype, actual.dtype());
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactActivationsMatchOfficialTransformers(DType dtype) throws IOException {
        compareReference(dtype);
    }

    static void compareReference(DType dtype) throws IOException {
        var directory = Qwen3Fixtures.path(dtype.name().toLowerCase(java.util.Locale.ROOT));
        var expected = Qwen3Json.read(directory.resolve("compact.json"));
        var model = Qwen3Pretrained.load(directory, dtype, true);
        try (var ignored = GradMode.noGrad()) {
            Tensor output = model.forwardBatch(Qwen3Fixtures.tokens(expected));
            assertEquals(dtype, output.dtype());
            assertArrayEquals(Qwen3Fixtures.floats(expected.get("logits")), output.toFloatArray(),
                    dtype == DType.FLOAT16 ? 0.0002f : 0.002f);
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void uniformModelsTrainWithCompactActivationsAndFloat32Gradients(DType dtype) {
        var model = new Qwen3(Qwen3ModelTest.tiny(), 3, ParameterFactory.uniform(dtype));
        assertEquals(dtype, model.forward(new int[]{1, 2, 3}).dtype());
        var batch = new Batch(new int[][]{{1, 2, 3}}, new int[][]{{2, 3, 4}});
        var trainer = CausalLMTraining.trainer(model, size -> batch, AdamW.defaultAdamW(0.005f));
        float first = trainer.trainStep(1), last = first;
        for (int step = 0; step < 9; step++) { last = trainer.trainStep(1); }
        assertTrue(last < first, first + " -> " + last);
        for (var parameter : model.parameters()) {
            assertEquals(dtype, parameter.value.dtype());
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            var gradients = Arrays.stream(ReferenceMath.doubles(parameter.grad().toFloatArray()));
            assertTrue(gradients.allMatch(Double::isFinite));
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void explicitCompactImportPreservesCachedOutputsWithoutChangingLegacyPrecision(DType dtype) throws IOException {
        var path = Qwen3Fixtures.path(dtype.name().toLowerCase(java.util.Locale.ROOT));
        var model = Qwen3Pretrained.load(path, dtype, true);
        int[] tokens = {1, 5, 7, 3};
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            assertEquals(DType.FLOAT32, Qwen3Pretrained.load(path, dtype).forward(tokens).dtype());
            for (int length = 1; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                Tensor cached = session.apply(prefix);
                assertEquals(dtype, cached.dtype());
                assertArrayEquals(model.forward(prefix).select(0, length - 1).toFloatArray(),
                        cached.toFloatArray(), dtype == DType.FLOAT16 ? 0.0003f : 0.002f);
            }
        }
    }
}
