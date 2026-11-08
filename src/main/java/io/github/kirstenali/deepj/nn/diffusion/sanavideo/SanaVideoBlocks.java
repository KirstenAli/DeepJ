// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelInference;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.io.UncheckedIOException;

/** Inference block streaming bounds parameter residency independently of existing model paths. */
final class SanaVideoBlocks implements AutoCloseable {
    private final HuggingFaceConfig config;
    private final TensorWeights shared;
    private final int channels;
    private final VideoWeightCache source;

    SanaVideoBlocks(HuggingFaceConfig config, TensorWeights shared, VideoWeightCache source) {
        this.config = config;
        this.shared = shared;
        this.source = source;
        channels = config.integer("num_attention_heads") * config.integer("attention_head_dim");
    }

    Tensor forward(Tensor input, Tensor caption, Tensor mask, Tensor time, VideoRotary rotary,
                   int layer, int[] shape) {
        ModelMemory.checkCancelled();
        if (source == null) return block(shared, input, caption, mask, time, rotary, layer, shape);
        if (GradMode.isEnabled()) throw new IllegalStateException("Streamed video blocks require GradMode.noGrad()");
        Tensor output = null;
        try (var pinned = ModelMemory.retain(input); var selected = weights(layer)) {
            output = evaluate(selected.weights(), input, caption, mask, time, rotary, layer, shape);
            output.retainDeviceBuffer();
            return output;
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        } finally {
            if (output != null) output.releaseDeviceBufferRetention();
        }
    }

    private VideoBlockCache.Selection weights(int layer) throws IOException {
        String prefix = "transformer_blocks." + layer + ".";
        return source.selectRepeated(prefix);
    }

    @Override public void close() { if (source != null) source.close(); }

    private Tensor evaluate(TensorWeights weights, Tensor input, Tensor caption, Tensor mask, Tensor time,
                            VideoRotary rotary, int layer, int[] shape) {
        if ((long) input.size(0) * input.size(1) > 512 || input.numel() > (1 << 20)) {
            return block(weights, input, caption, mask, time, rotary, layer, shape);
        }
        return ModelInference.evaluate(() -> block(weights, input, caption, mask, time, rotary, layer, shape));
    }

    private Tensor block(TensorWeights weights, Tensor input, Tensor caption, Tensor mask, Tensor time,
                         VideoRotary rotary, int layer, int[] shape) {
        String name = "transformer_blocks." + layer;
        var attention = new SanaVideoAttention(weights, config.integer("num_attention_heads"));
        try (var scope = TensorScope.open(); var inputPin = !GradMode.isEnabled() && input.size(1) > 8192
                ? ModelMemory.retain(input) : null) {
            Tensor parts = weights.tensor(name + ".scale_shift_table").unsqueeze(0)
                    .add(time.reshape(1, 6, channels));
            try (var fixed = ModelMemory.retain(parts)) {
                Tensor hidden = VideoMemory.finish(gated(input, attention.self(condition(input, parts, 0),
                        name + ".attn1", rotary), parts.slice(1, 2, 3)));
                try (var residual = ModelMemory.retain(hidden)) {
                    hidden = VideoMemory.finish(hidden.add(attention.cross(hidden, caption, mask, name + ".attn2")));
                }
                return scope.escape(feed(hidden, parts, weights, name, shape));
            }
        }
    }

    private Tensor feed(Tensor hidden, Tensor parts, TensorWeights weights, String name, int[] shape) {
        try (var scope = TensorScope.open(); var residual = ModelMemory.retain(hidden)) {
            Tensor feed = VideoLayout.volume(condition(hidden, parts, 3), shape[0], shape[1], shape[2]);
            Tensor output = gated(hidden, VideoLayout.tokens(SanaVideoFeedForward.apply(feed, weights,
                    name + ".ff")), parts.slice(1, 5, 6));
            return scope.escape(VideoMemory.finish(output));
        }
    }

    static Tensor gated(Tensor input, Tensor output, Tensor gate) {
        if (gate.dtype() == DType.FLOAT16) {
            return input.to(DType.FLOAT32).add(output.to(DType.FLOAT32).multiply(gate.to(DType.FLOAT32)));
        }
        return input.add(output.multiply(gate));
    }

    private Tensor condition(Tensor input, Tensor parts, int first) {
        return !GradMode.isEnabled() && input.size(1) > 8192
                ? VideoMemory.stage(() -> modulate(input, parts, first)) : modulate(input, parts, first);
    }

    private Tensor modulate(Tensor input, Tensor parts, int first) {
        Tensor norm = ModelOps.norm(input, config.decimal("norm_eps"));
        return norm.multiply(parts.slice(1, first + 1, first + 2).addScalar(1))
                .add(parts.slice(1, first, first + 1)).to(parts.dtype());
    }

}
