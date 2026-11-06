package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.function.Supplier;

/** Completes inference work while keeping large video activations on the selected backend. */
final class VideoMemory {
    private VideoMemory() {}

    static Tensor stage(Supplier<Tensor> operation) {
        Tensor output;
        try (var scope = TensorScope.open()) { output = scope.escape(operation.get()); }
        return finish(output);
    }

    static Tensor finish(Tensor output) {
        if (!GradMode.isEnabled()) {
            try (var pinned = ModelMemory.retain(output)) { Tensor.backend().releaseTemporaryResources(); }
        }
        return output;
    }
}
