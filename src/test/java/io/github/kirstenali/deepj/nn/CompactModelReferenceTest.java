package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.diffusion.sana.SanaTransformer;
import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoTransformer;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Backbone;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Config;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class CompactModelReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources");
    private static final DType[] TYPES = {DType.FLOAT16, DType.BFLOAT16};

    @AfterEach void release() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void gemmaUsesCompactActivationsAndMatchesTransformers(String backend) throws Exception {
        backend(backend);
        for (DType dtype : TYPES) {
            try (var weights = TensorWeights.open(ROOT.resolve("sana/gemma"), dtype);
                 var fixture = TensorWeights.open(ROOT.resolve("sana"), DType.FLOAT32);
                 var ignored = GradMode.noGrad()) {
                var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("sana/gemma/config.json")),
                        weights::tensor);
                Tensor result = model.forward(ids(fixture.tensor("gemma.tokens")), ids(fixture.tensor("gemma.mask")));
                compare("gemma", dtype, result);
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void sanaDenoiserMatchesCompactDiffusers(String backend) throws Exception {
        backend(backend);
        for (DType dtype : TYPES) {
            try (var weights = TensorWeights.open(ROOT.resolve("sana/transformer"), dtype);
                 var fixture = TensorWeights.open(ROOT.resolve("sana"), dtype); var ignored = GradMode.noGrad()) {
                var model = new SanaTransformer(ROOT.resolve("sana/transformer/config.json"), weights);
                compare("sana", dtype, model.forward(fixture.tensor("transformer.input"),
                        fixture.tensor("transformer.text"), ids(fixture.tensor("transformer.mask")), .6f, 4.5f));
            }
        }
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void videoDenoiserMatchesDiffusersWithLoadedAndStreamedCompactWeights(String backend) throws Exception {
        backend(backend);
        for (DType dtype : TYPES) {
            try (var weights = TensorWeights.open(ROOT.resolve("sana-video/transformer"), dtype);
                 var fixture = TensorWeights.open(ROOT.resolve("sana-video"), dtype);
                 var stream = SanaVideoTransformer.load(ROOT.resolve("sana-video/transformer"), 0, dtype);
                 var ignored = GradMode.noGrad()) {
                var model = new SanaVideoTransformer(ROOT.resolve("sana-video/transformer/config.json"), weights);
                Tensor x = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
                int[] mask = {1, 1, 1, 1, 1, 0, 0};
                compare("video", dtype, model.forward(x, text, mask, 733));
                compare("video", dtype, stream.forward(x, text, mask, 733));
            }
        }
    }

    private static void backend(String name) {
        if (name.equals("metal")) Assumptions.assumeTrue(MetalBackend.isAvailable());
        Tensor.setBackend(name.equals("metal") ? new MetalBackend() : new CpuBackend());
    }

    private static int[] ids(Tensor value) {
        int[] values = new int[value.numel()];
        float[] input = value.toFloatArray();
        for (int i = 0; i < input.length; i++) { values[i] = (int) input[i]; }
        return values;
    }

    private static void compare(String family, DType dtype, Tensor actual) throws Exception {
        try (var expected = TensorWeights.open(ROOT.resolve("compact-models"), DType.FLOAT32)) {
            assertEquals(dtype, actual.dtype());
            Tensor reference = expected.tensor(family + "." + dtype);
            assertArrayEquals(reference.shape(), actual.shape());
            float tolerance = dtype == DType.BFLOAT16 ? .04f : .006f;
            assertArrayEquals(reference.toFloatArray(), actual.toFloatArray(), tolerance, family + " " + dtype);
        }
    }
}
