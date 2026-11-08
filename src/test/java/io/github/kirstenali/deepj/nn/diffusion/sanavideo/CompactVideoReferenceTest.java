package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class CompactVideoReferenceTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");
    private static final Path MODEL = ROOT.resolve("pipeline");
    private final io.github.kirstenali.deepj.tensor.TensorBackend previous = Tensor.backend();

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(previous);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void threeStepFeedbackPreservesSamplerHistoryAcrossDenoiserCleanup(DType dtype) throws Exception {
        String name = dtype.name().toLowerCase(java.util.Locale.ROOT);
        try (var ignored = GradMode.noGrad();
             var fixture = TensorWeights.open(ROOT.resolve("compact-pipeline-reference"), DType.FLOAT32);
             var model = SanaVideoTransformer.load(MODEL.resolve("transformer"), 0, dtype);
             var solver = FlowDpmSolver.read(MODEL.resolve("scheduler/scheduler_config.json"), 3)) {
            Tensor sample = noise(), positive = fixture.tensor(name + ".positive");
            Tensor negative = fixture.tensor(name + ".negative");
            try (var fixed = ModelMemory.retain(positive, negative)) {
                for (int step = 0; step < 3; step++) {
                    sample = advance(model, solver, sample, positive, negative, dtype,
                            fixture.tensor(name + ".flow." + step));
                    assertArrayEquals(fixture.tensor(name + ".sample." + step).toFloatArray(),
                            sample.toFloatArray(), 2e-6f, name + " step " + step);
                }
            }
        }
    }

    private static Tensor advance(SanaVideoTransformer model, FlowDpmSolver solver, Tensor sample,
                                  Tensor positive, Tensor negative, DType dtype, Tensor referenceFlow) {
        int[] mask = new int[300];
        java.util.Arrays.fill(mask, 0, 7, 1);
        int[] empty = new int[300];
        empty[0] = 1;
        try (var pinned = ModelMemory.retain(sample)) {
            Tensor flow = VideoGuidance.predict(model, sample, positive, negative, mask, empty,
                    solver.timestep(), 6, dtype);
            assertArrayEquals(referenceFlow.toFloatArray(), flow.toFloatArray(),
                    predictionTolerance(dtype), "Same-input guided prediction");
            // Fixed Python flows isolate sampler history from projection rounding across feedback steps.
            return solver.step(sample, referenceFlow);
        }
    }

    private static float predictionTolerance(DType dtype) {
        // Classifier-free guidance magnifies the compact projection rounding differences by six.
        return switch (dtype) {
            case FLOAT32 -> 1e-4f;
            case FLOAT16 -> 0.04f;
            case BFLOAT16 -> 0.25f;
        };
    }

    private static Tensor noise() {
        float[] values = new float[16 * 3 * 2 * 2];
        var random = new java.util.Random(123);
        for (int index = 0; index < values.length; index++) { values[index] = (float) random.nextGaussian(); }
        return Tensor.from(values, 1, 16, 3, 2, 2);
    }

    @ParameterizedTest @EnumSource(DType.class)
    void threeStepPipelineMatchesPythonRenderedFrames(DType dtype) throws Exception {
        List<BufferedImage> expected;
        try (var fixture = TensorWeights.open(ROOT.resolve("compact-pipeline-reference"), DType.FLOAT32)) {
            expected = VideoFrames.rgb(fixture.tensor(dtype.name().toLowerCase(java.util.Locale.ROOT) + ".pixels"));
        }
        var options = new SanaVideo.Options(16, 16, 9, 3, 6, 123, dtype);
        compare(expected, new SanaVideo(MODEL).generate("a fox", options), dtype);
    }

    private static void compare(List<BufferedImage> expected, List<BufferedImage> actual, DType dtype) {
        assertEquals(expected.size(), actual.size());
        long total = 0;
        int maximum = 0;
        for (int frame = 0; frame < expected.size(); frame++) {
            for (int y = 0; y < 16; y++) {
                for (int x = 0; x < 16; x++) {
                    int[] error = pixel(expected.get(frame).getRGB(x, y), actual.get(frame).getRGB(x, y));
                    total += error[0];
                    maximum = Math.max(maximum, error[1]);
                }
            }
        }
        double mean = total / (9.0 * 16 * 16 * 3);
        System.out.printf(java.util.Locale.ROOT, "COMPACT VIDEO %s RGB max=%d mean=%.9g%n", dtype, maximum, mean);
        // BF16's rounding envelope is checked against the Python-only split-bias control in the audit report.
        int maxLimit = dtype == DType.BFLOAT16 ? 32 : 2;
        double meanLimit = dtype == DType.BFLOAT16 ? 1 : 0.1;
        assertTrue(maximum <= maxLimit && mean <= meanLimit, dtype + " RGB max=" + maximum + ", mean=" + mean);
    }

    private static int[] pixel(int expected, int actual) {
        int total = 0, maximum = 0;
        for (int shift = 0; shift <= 16; shift += 8) {
            int error = Math.abs((expected >> shift & 255) - (actual >> shift & 255));
            total += error;
            maximum = Math.max(maximum, error);
        }
        return new int[]{total, maximum};
    }
}
