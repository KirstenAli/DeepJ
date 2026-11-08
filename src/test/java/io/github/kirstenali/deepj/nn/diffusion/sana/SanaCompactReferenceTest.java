package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class SanaCompactReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana");

    private final TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactTransformerMatchesIndependentDiffusers(DType dtype) throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), dtype);
             var inputs = TensorWeights.open(ROOT, dtype); var reference = reference();
             var ignored = GradMode.noGrad()) {
            var model = new SanaTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor output = model.forward(inputs.tensor("transformer.input"), inputs.tensor("transformer.text"),
                    new int[]{1, 1, 1, 1, 1, 0, 0}, 0.6f, 4.5f);
            close(output, reference.tensor(label(dtype) + ".transformer"), dtype);
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactDecoderMatchesIndependentDiffusers(DType dtype) throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("vae"), dtype);
             var inputs = TensorWeights.open(ROOT, dtype); var reference = reference();
             var ignored = GradMode.noGrad()) {
            var model = new DcAeDecoder(ROOT.resolve("vae/config.json"), weights);
            for (int size : new int[]{2, 4}) {
                close(model.forward(inputs.tensor("vae" + size + ".input")),
                        reference.tensor(label(dtype) + ".vae" + size), dtype);
            }
        }
    }

    @Test void halfSelfAttentionClipsProjectionOverflowLikeDiffusers() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT16);
             var ignored = GradMode.noGrad()) {
            String prefix = "transformer_blocks.0.attn1";
            for (String name : new String[]{"to_q", "to_k", "to_v"}) {
                fill(weights, prefix + "." + name + ".weight", 1);
            }
            fill(weights, prefix + ".to_out.0.weight", 65504);
            var model = new SanaTransformer(ROOT.resolve("transformer/config.json"), weights);
            var method = SanaTransformer.class.getDeclaredMethod("selfAttention", Tensor.class, String.class);
            method.setAccessible(true);
            Tensor result = (Tensor) method.invoke(model, Tensor.ones(DType.FLOAT16, 1, 4, 16), prefix);
            for (float value : result.toFloatArray()) { assertEquals(65504, value); }
        }
    }

    private static void fill(TensorWeights weights, String name, float value) {
        Tensor target = weights.tensor(name);
        target.copyFrom(Tensor.ones(target.dtype(), target.shape()).multiplyScalar(value));
    }

    private static TensorWeights reference() throws Exception {
        return TensorWeights.open(ROOT.resolve("compact"), DType.FLOAT32);
    }

    private static String label(DType dtype) { return dtype.name().toLowerCase(Locale.ROOT); }

    private static void close(Tensor actual, Tensor expected, DType dtype) {
        assertArrayEquals(expected.shape(), actual.shape());
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), dtype == DType.FLOAT16 ? 0.01f : 0.08f);
    }
}
