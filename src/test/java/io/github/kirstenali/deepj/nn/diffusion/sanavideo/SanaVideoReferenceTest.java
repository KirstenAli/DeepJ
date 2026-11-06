package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaVideoReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void transformerMatchesDiffusersWithRotaryMotionAndMaskedCaptions() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new SanaVideoTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor output = model.forward(fixture.tensor("transformer.input"), fixture.tensor("transformer.text"),
                    new int[]{1, 1, 1, 1, 1, 0, 0}, 733);
            close(output, fixture.tensor("transformer.output"), 8e-5f);
        }
    }

    @Test void wanDecoderMatchesFirstFrameAndSubsequentTemporalChunks() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("vae"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new WanVideoDecoder(ROOT.resolve("vae/config.json"), weights);
            for (int time = 1; time <= 4; time++) {
                Tensor output = model.forward(fixture.tensor("vae" + time + ".input"));
                assertEquals(1 + 4 * (time - 1), output.size(2));
                close(output, fixture.tensor("vae" + time + ".output"), 2e-4f);
            }
        }
    }

    @Test void streamedBlocksMatchTheSameReferenceWhileReleasingTheirWeights() throws Exception {
        try (var ignored = GradMode.noGrad(); var fixture = TensorWeights.open(ROOT, DType.FLOAT32);
             var model = SanaVideoTransformer.streamed(ROOT.resolve("transformer"))) {
            Tensor output = model.forward(fixture.tensor("transformer.input"), fixture.tensor("transformer.text"),
                    new int[]{1, 1, 1, 1, 1, 0, 0}, 733);
            close(output, fixture.tensor("transformer.output"), 8e-5f);
        }
    }

    @Test void videoTensorEntryPointsRequireFloat32Inputs() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT32);
             var vae = TensorWeights.open(ROOT.resolve("vae"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            var model = new SanaVideoTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor latent = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
            int[] valid = {1, 1, 1, 1, 1, 0, 0};
            assertThrows(IllegalArgumentException.class,
                    () -> model.forward(latent.to(DType.BFLOAT16), text, valid, 733));
            assertThrows(IllegalArgumentException.class,
                    () -> model.forward(latent, text.to(DType.FLOAT16), valid, 733));
            var decoder = new WanVideoDecoder(ROOT.resolve("vae/config.json"), vae);
            assertThrows(IllegalArgumentException.class,
                    () -> decoder.forward(fixture.tensor("vae1.input").to(DType.BFLOAT16)));
        }
    }

    @Test void convolutionOutputsAndGradientsMatchTorch() throws Exception {
        try (var fixture = TensorWeights.open(ROOT, DType.FLOAT32)) {
            for (String prefix : new String[]{"conv", "point", "patch"}) {
                convolution(fixture, prefix);
            }
        }
    }

    @Test void transformerInputCaptionAndVideoWeightsHaveReferenceGradients() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("transformer"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32)) {
            var model = new SanaVideoTransformer(ROOT.resolve("transformer/config.json"), weights);
            Tensor latent = fixture.tensor("transformer.input"), text = fixture.tensor("transformer.text");
            Tensor output = model.forward(latent, text, new int[]{1, 1, 1, 1, 1, 0, 0}, 733);
            output.multiply(output).sumTensor().divideScalar(output.numel()).backward();
            close(latent.grad(), fixture.tensor("transformer.dx"), 5e-5f);
            close(text.grad(), fixture.tensor("transformer.dt"), 5e-5f);
            for (String name : new String[]{"patch_embedding.weight", "transformer_blocks.0.ff.conv_temp.weight",
                    "transformer_blocks.0.attn1.to_q.weight"}) {
                close(weights.tensor(name).grad(), fixture.tensor("transformer.grad." + name), 5e-5f);
            }
        }
    }

    @Test void decoderGradientsCrossTheCausalCacheAndTemporalUpsamplers() throws Exception {
        try (var weights = TensorWeights.open(ROOT.resolve("vae"), DType.FLOAT32);
             var fixture = TensorWeights.open(ROOT, DType.FLOAT32)) {
            var model = new WanVideoDecoder(ROOT.resolve("vae/config.json"), weights);
            Tensor latent = fixture.tensor("vae2.input");
            Tensor output = model.forward(latent);
            output.multiply(output).sumTensor().divideScalar(output.numel()).backward();
            close(latent.grad(), fixture.tensor("vae.dx"), 3e-4f);
            close(weights.tensor("decoder.conv_in.weight").grad(), fixture.tensor("vae.dw"), 3e-4f);
        }
    }

    private static void convolution(TensorWeights fixture, String name) {
        Tensor input = fixture.tensor(name + ".input"), weight = fixture.tensor(name + ".weight");
        Tensor bias = fixture.tensor(name + ".bias");
        Tensor output = name.equals("patch") ? VideoLayout.volume(VideoConvolution.patch(input, weight, bias),
                input.size(2), input.size(3) / 2, input.size(4) / 2)
                : VideoConvolution.causal(input, weight, bias, null);
        close(output, fixture.tensor(name + ".output"), 1e-5f);
        output.multiply(output).sumTensor().backward();
        close(input.grad(), fixture.tensor(name + ".dx"), 5e-4f);
        close(weight.grad(), fixture.tensor(name + ".dw"), 5e-4f);
        close(bias.grad(), fixture.tensor(name + ".db"), 5e-4f);
    }

    @Test void flowSolverMatchesEveryDiffusersStepIncludingFinalOrderReduction() throws Exception {
        try (var fixture = TensorWeights.open(ROOT, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            for (int steps : new int[]{1, 2, 5, 15, 20, 50}) {
                solver(fixture, steps);
            }
        }
    }

    private static void solver(TensorWeights fixture, int steps) throws Exception {
        String name = "solver" + steps;
        Tensor sample = fixture.tensor(name + ".input");
        float[] times = fixture.tensor(name + ".times").toFloatArray();
        try (var solver = FlowDpmSolver.read(ROOT.resolve("scheduler/scheduler_config.json"), steps)) {
            for (int i = 0; i < steps; i++) {
                assertEquals((int) times[i], solver.timestep());
                sample = solver.step(sample, fixture.tensor(name + "." + i + ".flow"));
                close(sample, fixture.tensor(name + "." + i + ".output"), 2e-5f);
            }
            assertThrows(IllegalStateException.class, solver::timestep);
        }
    }

    private static void close(Tensor actual, Tensor expected, float tolerance) {
        assertAll(() -> assertArrayEquals(expected.shape(), actual.shape()),
                () -> assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), tolerance));
    }
}
