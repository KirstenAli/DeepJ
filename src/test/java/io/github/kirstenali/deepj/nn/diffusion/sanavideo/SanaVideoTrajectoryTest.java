package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SanaVideoTrajectoryTest {
    private static final Path ROOT = Path.of("src/test/resources/sana-video");
    private static final int[] POSITIVE_MASK = {1, 1, 1, 1, 1, 0, 0};
    private static final int[] NEGATIVE_MASK = {1, 1, 0, 0, 0, 0, 0};

    @BeforeEach void backend() {
        Tensor.setBackend(Boolean.getBoolean("sanavideo.test.metal") ? new MetalBackend() : new CpuBackend());
    }

    @Test void fullPrecisionFeedbackAndDecodedFramesMatchPython() throws Exception {
        trajectory(20);
        trajectory(50);
    }

    private static void trajectory(int steps) throws Exception {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open();
             var fixture = TensorWeights.open(ROOT.resolve("trajectory"), DType.FLOAT32);
             var model = SanaVideoTransformer.streamed(ROOT.resolve("transformer"));
             var solver = FlowDpmSolver.read(ROOT.resolve("scheduler/scheduler_config.json"), steps)) {
            Tensor sample = fixture.tensor("input");
            Tensor positive = fixture.tensor("positive"), negative = fixture.tensor("negative");
            String name = "float32." + steps;
            try (var pinned = ModelMemory.retain(positive, negative)) {
                for (int step = 0; step < steps; step++) {
                    sample = advance(model, solver, sample, positive, negative);
                    compare(name + "." + step, sample, fixture.tensor(name + "." + step + ".sample"), 1e-4f);
                }
                decode(sample, fixture.tensor(name + ".pixels"), name, 1e-4f);
            }
        }
    }

    private static Tensor advance(SanaVideoTransformer model, FlowDpmSolver solver, Tensor sample,
                                  Tensor positive, Tensor negative) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(sample)) {
            Tensor input = sample;
            Tensor conditioned = model.forward(input, positive, POSITIVE_MASK, solver.timestep());
            try (var fixed = ModelMemory.retain(conditioned)) {
                Tensor unconditioned = model.forward(input, negative, NEGATIVE_MASK, solver.timestep());
                Tensor flow = unconditioned.add(conditioned.subtract(unconditioned).multiplyScalar(6));
                return scope.escape(solver.step(sample, flow));
            }
        }
    }

    private static void decode(Tensor sample, Tensor expected, String name, float tolerance) throws Exception {
        try (var pinned = ModelMemory.retain(sample, expected);
             var weights = TensorWeights.open(ROOT.resolve("vae"), DType.FLOAT32)) {
            var decoder = new WanVideoDecoder(ROOT.resolve("vae/config.json"), weights);
            compare(name + ".pixels", decoder.forward(sample), expected, tolerance);
        }
    }

    private static void compare(String name, Tensor actual, Tensor expected, float tolerance) {
        assertArrayEquals(expected.shape(), actual.shape(), name);
        float[] values = actual.toFloatArray(), reference = expected.toFloatArray();
        double maximum = 0, squared = 0;
        for (int index = 0; index < values.length; index++) {
            double error = Math.abs(values[index] - reference[index]);
            maximum = Math.max(maximum, error);
            squared += error * error;
        }
        System.out.printf(java.util.Locale.ROOT, "PARITY %s max=%.9g rms=%.9g%n",
                name, maximum, Math.sqrt(squared / values.length));
        assertTrue(Double.isFinite(maximum) && maximum <= tolerance, name + " maximum error: " + maximum);
    }
}
