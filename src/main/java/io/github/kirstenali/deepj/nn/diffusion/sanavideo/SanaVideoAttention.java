// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorScope;

final class SanaVideoAttention {
    private final TensorWeights weights;
    private final int heads;

    SanaVideoAttention(TensorWeights weights, int heads) {
        this.weights = weights;
        this.heads = heads;
    }

    Tensor self(Tensor input, String name, VideoRotary rotary) {
        if (!GradMode.isEnabled() && input.size(1) > 8192) return stagedSelf(input, name, rotary);
        Tensor q = project(input, name, "q").reluActivation();
        Tensor k = project(input, name, "k").reluActivation();
        Tensor v = ModelOps.heads(ModelOps.linear(input, weights, name + ".to_v"), heads);
        return selfValues(q, k, v, name, rotary);
    }

    private Tensor stagedSelf(Tensor input, String name, VideoRotary rotary) {
        try (var scope = TensorScope.open(); var fixed = ModelMemory.retain(input)) {
            Tensor q = VideoMemory.stage(() -> project(input, name, "q").reluActivation());
            try (var query = ModelMemory.retain(q)) {
                Tensor k = VideoMemory.stage(() -> project(input, name, "k").reluActivation());
                try (var key = ModelMemory.retain(k)) {
                    Tensor v = VideoMemory.stage(() -> ModelOps.heads(
                            ModelOps.linear(input, weights, name + ".to_v"), heads));
                    return scope.escape(stagedValues(q, k, v, name, rotary));
                }
            }
        }
    }

    private Tensor stagedValues(Tensor q, Tensor k, Tensor v, String name, VideoRotary rotary) {
        Tensor numerator = stagedNumerator(q, k, v, rotary);
        try (var fixed = ModelMemory.retain(numerator)) {
            Tensor divisor = VideoMemory.stage(() -> k.sum(-2, true).matmul(q.transpose(-1, -2)).addScalar(1e-15f));
            Tensor output = VideoMemory.stage(() -> numerator.multiply(Tensor.from(new float[]{1}, 1)
                    .divide(divisor).to(divisor.dtype())).transpose(-1, -2));
            return VideoMemory.stage(() -> finish(output, name));
        }
    }

    private Tensor stagedNumerator(Tensor q, Tensor k, Tensor v, VideoRotary rotary) {
        try (var scope = TensorScope.open(); var value = ModelMemory.retain(v)) {
            Tensor qr = VideoMemory.stage(() -> rotary.apply(q).to(DType.FLOAT32));
            try (var query = ModelMemory.retain(qr)) {
                Tensor kr = VideoMemory.stage(() -> rotary.apply(k).to(DType.FLOAT32));
                try (var key = ModelMemory.retain(kr)) {
                    return scope.escape(VideoMemory.stage(() -> v.to(DType.FLOAT32).transpose(-1, -2)
                            .matmul(kr).matmul(qr.transpose(-1, -2))));
                }
            }
        }
    }

    private Tensor selfValues(Tensor q, Tensor k, Tensor v, String name, VideoRotary rotary) {
        Tensor qr = rotary.apply(q).to(DType.FLOAT32), kr = rotary.apply(k).to(DType.FLOAT32);
        // Keep the upstream multiplication layout to limit rounding drift across denoising steps.
        Tensor numerator = v.to(DType.FLOAT32).transpose(-1, -2).matmul(kr).matmul(qr.transpose(-1, -2));
        Tensor divisor = k.sum(-2, true).matmul(q.transpose(-1, -2)).addScalar(1e-15f);
        Tensor reciprocal = Tensor.from(new float[]{1}, 1).divide(divisor).to(divisor.dtype());
        return finish(numerator.multiply(reciprocal).transpose(-1, -2), name);
    }

    Tensor cross(Tensor input, Tensor caption, Tensor mask, String name) {
        if (!GradMode.isEnabled() && input.size(1) > 8192) return stagedCross(input, caption, mask, name);
        Tensor q = project(input, name, "q"), k = project(caption, name, "k");
        Tensor v = ModelOps.heads(ModelOps.linear(caption, weights, name + ".to_v"), heads);
        return crossValues(q, k, v, mask, name);
    }

    private Tensor stagedCross(Tensor input, Tensor caption, Tensor mask, String name) {
        try (var scope = TensorScope.open(); var fixed = ModelMemory.retain(input, caption)) {
            Tensor q = VideoMemory.stage(() -> project(input, name, "q"));
            try (var query = ModelMemory.retain(q)) {
                Tensor k = project(caption, name, "k");
                Tensor v = ModelOps.heads(ModelOps.linear(caption, weights, name + ".to_v"), heads);
                return scope.escape(crossValues(q, k, v, mask, name));
            }
        }
    }

    private Tensor crossValues(Tensor q, Tensor k, Tensor v, Tensor mask, String name) {
        Tensor output = VideoCrossAttention.apply(q, k, v, mask, (float) (1 / Math.sqrt(q.size(-1))));
        return finish(output, name);
    }

    private Tensor project(Tensor input, String name, String kind) {
        Tensor projected = ModelOps.linear(input, weights, name + ".to_" + kind);
        return ModelOps.heads(ModelOps.rms(projected, weights.tensor(name + ".norm_" + kind + ".weight"),
                null, 1e-5f, false), heads);
    }

    private Tensor finish(Tensor input, String name) {
        return ModelOps.linear(ModelOps.mergeHeads(input).to(weights.tensor(name + ".to_out.0.weight").dtype()),
                weights, name + ".to_out.0");
    }
}
