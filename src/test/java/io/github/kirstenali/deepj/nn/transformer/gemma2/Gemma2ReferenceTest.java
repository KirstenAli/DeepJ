package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2ReferenceTest {
    static final Path ROOT = Path.of("src/test/resources/gemma2");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("gemma.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT32", "BFLOAT16"})
    void fullLogitsMatchIndependentTransformers(DType dtype) throws Exception {
        try (var model = Gemma2Pretrained.load(ROOT, dtype); var reference = reference();
             var ignored = GradMode.noGrad()) {
            String name = dtype == DType.FLOAT32 ? "logits" : "logits_bf16";
            assertArrayEquals(reference.tensor(name).toFloatArray(),
                    model.forward(ids(reference.tensor("tokens"))).toFloatArray(),
                    dtype == DType.FLOAT32 ? 3e-5f : 0.008f);
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT32", "BFLOAT16"})
    void cachedDecodingMatchesFullLogitsAcrossTheLocalAttentionBoundary(DType dtype) throws Exception {
        try (var model = Gemma2Pretrained.load(ROOT, dtype); var reference = reference();
             var session = model.openGeneration()) {
            int[] tokens = ids(reference.tensor("tokens"));
            for (int length = 1; length <= tokens.length; length++) {
                Tensor row = session.apply(Arrays.copyOf(tokens, length));
                String name = dtype == DType.FLOAT32 ? "logits" : "logits_bf16";
                float[] expected = reference.tensor(name).reshape(tokens.length, model.config().vocabSize())
                        .rowData(length - 1);
                assertArrayEquals(expected, row.toFloatArray(), dtype == DType.FLOAT32 ? 4e-5f : 0.008f);
                Tensor.backend().releaseTemporaryResources();
            }
        }
    }

    @Test void growingTheCacheAndChangingThePromptRebuildsCorrectly() throws Exception {
        try (var model = Gemma2Pretrained.load(ROOT, DType.FLOAT32); var reference = reference();
             var session = model.openGeneration()) {
            int[] tokens = ids(reference.tensor("long_tokens"));
            session.apply(Arrays.copyOf(tokens, 128)).toFloatArray();
            Tensor.backend().releaseTemporaryResources();
            assertArrayEquals(reference.tensor("long_logits").toFloatArray(),
                    session.apply(tokens).toFloatArray(), 4e-5f);
            int[] shortPrompt = ids(reference.tensor("tokens"));
            assertArrayEquals(reference.tensor("logits").reshape(shortPrompt.length, model.config().vocabSize())
                    .rowData(shortPrompt.length - 1), session.apply(shortPrompt).toFloatArray(), 4e-5f);
        }
    }

    @Test void embeddingGradientsMatchTransformersIncludingTheTiedOutputHead() throws Exception {
        try (var model = Gemma2Pretrained.load(ROOT, DType.FLOAT32); var reference = reference()) {
            Tensor logits = model.forward(ids(reference.tensor("tokens")));
            logits.multiply(logits).sumTensor().divideScalar(logits.numel()).backward();
            Tensor gradient = model.parameters().stream().filter(p -> p.name().equals("model.embed_tokens.weight"))
                    .findFirst().orElseThrow().value.grad();
            assertArrayEquals(reference.tensor("embedding_gradient").toFloatArray(), gradient.toFloatArray(), 2e-5f);
        }
    }

    @Test void sessionsRejectWeightMutationAndUnsupportedRollingMode() throws Exception {
        try (var model = Gemma2Pretrained.load(ROOT, DType.FLOAT32); var session = model.openGeneration();
             var ignored = GradMode.noGrad()) {
            model.parameters().get(0).value.multiplyScalarInPlace(1);
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{2}));
            assertThrows(UnsupportedOperationException.class, () -> model.openGeneration(ContextWindowMode.ROLLING));
        }
    }

    private static TensorWeights reference() throws Exception {
        return TensorWeights.open(ROOT.resolve("oracles"), DType.FLOAT32);
    }

    private static int[] ids(Tensor input) {
        float[] values = input.toFloatArray();
        int[] result = new int[values.length];
        for (int index = 0; index < values.length; index++) { result[index] = (int) values[index]; }
        return result;
    }
}
