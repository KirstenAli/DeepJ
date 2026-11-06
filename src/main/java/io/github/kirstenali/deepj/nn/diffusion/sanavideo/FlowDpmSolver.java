// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** DPM-Solver++ order-two midpoint for flow predictions, with shifted flow sigmas and a zero final sigma. */
public final class FlowDpmSolver implements AutoCloseable {
    private final float[] sigmas;
    private final int[] times;
    private int index;
    private Tensor previous;

    public FlowDpmSolver(int steps, float shift) {
        if (steps < 1 || steps > 1000 || !Float.isFinite(shift) || shift <= 0) {
            throw new IllegalArgumentException("Expected 1-1000 steps and a positive finite flow shift");
        }
        sigmas = new float[steps + 1];
        times = new int[steps];
        for (int i = 0; i < steps; i++) {
            double sigma = 0.999 * (steps - i) / steps;
            sigma = shift * sigma / (1 + (shift - 1) * sigma);
            sigmas[i] = (float) sigma;
            times[i] = (int) (sigma * 1000);
        }
    }

    public static FlowDpmSolver read(Path path, int steps) throws IOException {
        // Diffusers writes Python's -Infinity extension for the inactive lambda clipping setting.
        var config = Qwen3Json.object(Qwen3Json.parse(Files.readString(path).replace("-Infinity", "null")));
        for (String[] pair : new String[][]{{"_class_name", "DPMSolverMultistepScheduler"},
                {"algorithm_type", "dpmsolver++"}, {"prediction_type", "flow_prediction"},
                {"solver_type", "midpoint"}, {"final_sigmas_type", "zero"}}) {
            if (!pair[1].equals(config.get(pair[0]))) throw new IOException("Unsupported scheduler " + pair[0]);
        }
        if (!Boolean.TRUE.equals(config.get("use_flow_sigmas")) || !Long.valueOf(2).equals(config.get("solver_order"))
                || !Long.valueOf(1000).equals(config.get("num_train_timesteps"))
                || Boolean.TRUE.equals(config.get("thresholding"))
                || Boolean.TRUE.equals(config.get("use_dynamic_shifting"))) {
            throw new IOException("Expected the SANA-Video flow scheduler configuration");
        }
        return new FlowDpmSolver(steps, ((Number) config.get("flow_shift")).floatValue());
    }

    public int timestep() {
        if (index == times.length) throw new IllegalStateException("Video sampling is complete");
        return times[index];
    }

    public Tensor step(Tensor sample, Tensor flow) {
        timestep();
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(sample, flow)) {
            Tensor clean = sample.subtract(flow.multiplyScalar(sigmas[index]));
            float ratio = sigmas[index + 1] / sigmas[index];
            Tensor estimate = previous == null || index == times.length - 1 ? clean : midpoint(clean);
            Tensor next = sample.multiplyScalar(ratio).add(estimate.multiplyScalar(1 - ratio));
            clean.materialize();
            clean.retainDeviceBuffer();
            if (previous != null) previous.releaseDeviceBufferRetention();
            previous = clean;
            index++;
            return scope.escape(ModelMemory.finish(next));
        }
    }

    private Tensor midpoint(Tensor clean) {
        double step = lambda(sigmas[index + 1]) - lambda(sigmas[index]);
        double last = lambda(sigmas[index]) - lambda(sigmas[index - 1]);
        return clean.add(clean.subtract(previous).multiplyScalar((float) (0.5 * step / last)));
    }

    private static double lambda(float sigma) { return Math.log1p(-sigma) - Math.log(sigma); }

    @Override public void close() {
        if (previous != null) previous.releaseDeviceBufferRetention();
        previous = null;
    }
}
