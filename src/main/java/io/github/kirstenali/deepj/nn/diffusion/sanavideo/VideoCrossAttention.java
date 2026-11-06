package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.ArrayList;

/** Bounds inference score matrices by splitting independent query rows; key/value context stays complete. */
final class VideoCrossAttention {
    private VideoCrossAttention() {}

    static Tensor apply(Tensor query, Tensor key, Tensor value, Tensor mask, float scale) {
        if (GradMode.isEnabled() || query.size(2) <= 8192) {
            return ModelOps.attention(query, key, value, mask, scale, 0);
        }
        var results = new ArrayList<Tensor>();
        var retained = new ArrayList<ModelMemory.Pin>();
        try (var scope = TensorScope.open(); var inputs = ModelMemory.retain(query, key, value);
             var bias = mask == null ? null : ModelMemory.retain(mask)) {
            for (int start = 0; start < query.size(2); start += 1024) {
                ModelMemory.checkCancelled();
                Tensor part = query.slice(2, start, Math.min(query.size(2), start + 1024));
                Tensor output = VideoMemory.finish(ModelOps.attention(part, key, value, mask, scale, 0));
                retained.add(ModelMemory.retain(output));
                results.add(output);
            }
            return scope.escape(VideoMemory.finish(ModelOps.cat(2, results.toArray(Tensor[]::new))));
        } finally { retained.forEach(ModelMemory.Pin::close); }
    }
}
