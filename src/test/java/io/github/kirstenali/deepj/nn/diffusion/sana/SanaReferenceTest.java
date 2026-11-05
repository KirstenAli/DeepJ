package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Backbone;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Config;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.GroupedConv2D;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void gemmaMatchesTransformersIncludingSlidingWindowAndPadding() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("gemma"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("gemma/config.json")), weights::tensor);
            Tensor output = model.forward(integers(fixture.tensor("gemma.tokens")),
                    integers(fixture.tensor("gemma.mask")));
            close(output, fixture.tensor("gemma.output"), 5e-5f);
        }
    }

    @Test void bf16GemmaMatchesTransformers() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("gemma"), DType.BFLOAT16);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new Gemma2Backbone(Gemma2Config.read(ROOT.resolve("gemma/config.json")), weights::tensor);
            Tensor output = model.forward(integers(fixture.tensor("gemma.tokens")),
                    integers(fixture.tensor("gemma.mask")));
            close(output, fixture.tensor("gemma.output_bf16"), 0.035f);
        }
    }

    @Test void sanaTransformerMatchesDiffusersWithConditioningAndMaskedCrossAttention() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new SanaTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor output = model.forward(fixture.tensor("transformer.input"), fixture.tensor("transformer.text"),
                    integers(fixture.tensor("transformer.mask")), 0.6f, 4.5f);
            close(output, fixture.tensor("transformer.output"), 5e-5f);
        }
    }

    @Test void dcAeMatchesDiffusersForQuadraticAndLinearReferenceAttention() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("vae"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new DcAeDecoder(ROOT.resolve("vae/config.json"), weights);
            for (int size : new int[]{2, 4}) {
                close(model.forward(fixture.tensor("vae" + size + ".input")),
                        fixture.tensor("vae" + size + ".output"), 5e-5f);
            }
        }
    }

    @Test void groupedConvolutionAndAllGradientsMatchTorch() throws Exception {
        try (var fixture = TensorWeights.open(ROOT, DType.FLOAT32)) {
            Tensor input = fixture.tensor("conv.input"), weight = fixture.tensor("conv.weight");
            Tensor bias = fixture.tensor("conv.bias");
            Tensor output = GroupedConv2D.apply(input, weight, bias, 2);
            close(output, fixture.tensor("conv.output"), 1e-5f);
            output.multiply(output).sumTensor().backward();
            close(input.grad(), fixture.tensor("conv.dx"), 2e-4f);
            close(weight.grad(), fixture.tensor("conv.dw"), 2e-4f);
            close(bias.grad(), fixture.tensor("conv.db"), 2e-4f);
        }
    }

    @Test void convolutionTilesPreserveBothSidesOfAnInternalBoundary() {
        try (var ignored = GradMode.noGrad()) {
            Tensor output = GroupedConv2D.apply(Tensor.ones(DType.FLOAT32, 1, 512, 4, 512),
                    Tensor.ones(DType.FLOAT32, 512, 1, 3, 3), null, 512);
            float[] values = output.toFloatArray();
            assertEquals(4, values[0]);
            assertEquals(9, values[2 * 512 + 256]);
            assertEquals(6, values[3 * 512 + 256]);
            assertEquals(4, values[values.length - 1]);
        }
    }

    @Test void geluRemainsFiniteAcrossPretrainedActivationMagnitudes() {
        float[] values = {-1000, -200, -20, -5, -1, 0, 1, 5, 20, 200, 1000};
        float[] expected = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            double x = values[i];
            expected[i] = (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI)
                    * (x + 0.044715 * x * x * x))));
        }
        Tensor input = Tensor.from(values, values.length).requiresGrad(true);
        Tensor output = input.geluActivation();
        assertArrayEquals(expected, output.toFloatArray(), 1e-5f);
        output.sumTensor().backward();
        for (float gradient : input.grad().toFloatArray()) { assertTrue(Float.isFinite(gradient)); }
        assertThrows(IllegalStateException.class,
                () -> ModelMemory.finite(Tensor.from(new float[]{Float.NaN}, 1), "test"));
    }

    @Test void imageAndOptionsValidateContract() {
        assertThrows(IllegalArgumentException.class, () -> new SanaSprint.Options(31, 32, 1, 4.5f, 1));
        assertThrows(IllegalArgumentException.class, () -> SanaScm.schedule(5));
        var image = SanaImages.rgb(Tensor.from(new float[]{-1, 0, 1}, 1, 3, 1, 1));
        assertEquals(0x0080ff, image.getRGB(0, 0) & 0xffffff);
    }

    private static int[] integers(Tensor tensor) {
        float[] values = tensor.toFloatArray();
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) { result[i] = (int) values[i]; }
        return result;
    }

    private static void close(Tensor actual, Tensor expected, float tolerance) {
        assertArrayEquals(expected.shape(), actual.shape());
        float[] a = actual.toFloatArray(), b = expected.toFloatArray();
        float error = 0;
        for (int i = 0; i < a.length; i++) { error = Math.max(error, Math.abs(a[i] - b[i])); }
        assertTrue(Float.isFinite(error) && error <= tolerance, "max error " + error + " vs " + tolerance
                + "; actual " + Arrays.toString(Arrays.copyOf(a, Math.min(8, a.length)))
                + "; expected " + Arrays.toString(Arrays.copyOf(b, Math.min(8, b.length))));
    }
}
