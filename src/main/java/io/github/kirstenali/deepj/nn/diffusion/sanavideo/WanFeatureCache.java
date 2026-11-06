// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Per-decode causal state. No state is shared between videos or stored in model parameters. */
final class WanFeatureCache implements AutoCloseable {
    private final Map<String, Tensor> history = new HashMap<>();
    private final Set<String> temporal = new HashSet<>();
    private final TensorWeights weights;

    WanFeatureCache(TensorWeights weights) { this.weights = weights; }

    Tensor convolution(Tensor input, String name) { return convolution(input, name, false); }

    private Tensor convolution(Tensor input, String name, boolean padHistory) {
        Tensor kernel = weights.tensor(name + ".weight");
        Tensor bias = weights.contains(name + ".bias") ? weights.tensor(name + ".bias") : null;
        Tensor previous = history.get(name);
        Tensor result = VideoConvolution.causal(input, kernel, bias, previous);
        if (kernel.size(2) > 1) remember(name, input, previous, kernel.size(2) - 1, padHistory);
        return VideoMemory.finish(result);
    }

    private void remember(String name, Tensor input, Tensor previous, int count, boolean pad) {
        Tensor tail = tail(input, previous, count, pad);
        if (GradMode.isEnabled()) tail.materialize();
        tail.retainDeviceBuffer();
        history.put(name, tail);
        if (previous != null) previous.releaseDeviceBufferRetention();
    }

    private static Tensor tail(Tensor input, Tensor previous, int count, boolean pad) {
        Tensor recent = input.slice(2, Math.max(0, input.size(2) - count), input.size(2));
        int missing = count - recent.size(2);
        if (missing > 0 && previous != null) {
            Tensor prefix = previous.slice(2, Math.max(0, previous.size(2) - missing), previous.size(2));
            return ModelOps.cat(2, prefix, recent);
        }
        if (missing > 0 && pad) {
            Tensor zero = Tensor.backend().zeros(input.dtype(), input.size(0), input.size(1),
                    missing, input.size(3), input.size(4));
            return ModelOps.cat(2, zero, recent);
        }
        return recent.copy();
    }

    Tensor upsampleTime(Tensor input, String name) {
        if (temporal.add(name)) return input;
        Tensor expanded = convolution(input, name, true);
        return expanded.reshape(input.size(0), 2, input.size(1), input.size(2), input.size(3), input.size(4))
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(input.size(0), input.size(1), input.size(2) * 2, input.size(3), input.size(4));
    }

    @Override public void close() {
        history.values().forEach(Tensor::releaseDeviceBufferRetention);
        history.clear();
        temporal.clear();
    }
}
