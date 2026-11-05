package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaAutogradTest {
    private static final Path ROOT = Path.of("src/test/resources/sana");

    @Test void transformerInputAndParameterGradientsFollowDeepjAutograd() throws Exception {
        Tensor.setBackend(Boolean.getBoolean("sana.test.metal") ? new MetalBackend() : new CpuBackend());
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32)) {
            var model = new SanaTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor input = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
            loss(model, input, text).backward();
            float gradient = input.grad().toFloatArray()[0];
            float estimate = difference(model, input, text);
            assertEquals(estimate, gradient, 0.003f);
            assertNotNull(weights.tensor("patch_embed.proj.weight").grad());
        }
    }

    private static Tensor loss(SanaTransformer model, Tensor input, Tensor text) {
        Tensor output = model.forward(input, text, new int[]{1, 1, 1, 1, 1, 0, 0}, 0.6f, 4.5f);
        return output.multiply(output).sumTensor();
    }

    private static float difference(SanaTransformer model, Tensor input, Tensor text) {
        try (var ignored = GradMode.noGrad()) {
            float[] values = input.toFloatArray();
            values[0] += 0.002f;
            float plus = loss(model, Tensor.from(values, input.shape()), text).item();
            values[0] -= 0.004f;
            float minus = loss(model, Tensor.from(values, input.shape()), text).item();
            return (plus - minus) / 0.004f;
        }
    }
}
