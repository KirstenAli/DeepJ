// Portions adapted from Diffusers (Apache-2.0); see META-INF/licenses/sana-video-NOTICE.txt.
package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.tensor.Tensor;

final class SanaVideoConditioning {
    private final TensorWeights weights;

    SanaVideoConditioning(TensorWeights weights) { this.weights = weights; }

    Tensor time(float timestep) {
        float[] values = new float[256];
        for (int pair = 0; pair < 128; pair++) {
            double phase = timestep / Math.pow(10000, pair / 128.0);
            values[pair] = (float) Math.cos(phase);
            values[pair + 128] = (float) Math.sin(phase);
        }
        String name = "time_embed.emb.timestep_embedder";
        Tensor first = ModelOps.linear(Tensor.from(values, 1, 256)
                .to(weights.tensor(name + ".linear_1.weight").dtype()), weights, name + ".linear_1");
        return ModelOps.linear(ModelOps.silu(first), weights, name + ".linear_2");
    }

    Tensor modulation(Tensor embedding) {
        return ModelOps.linear(ModelOps.silu(embedding), weights, "time_embed.linear");
    }

    Tensor caption(Tensor text) {
        Tensor projected = ModelOps.linear(text, weights, "caption_projection.linear_1").geluActivation();
        Tensor result = ModelOps.linear(projected, weights, "caption_projection.linear_2");
        return ModelOps.rms(result, weights.tensor("caption_norm.weight"), null, 1e-5f, false);
    }

    static Tensor mask(int[] valid) {
        float[] bias = new float[valid.length];
        for (int token = 0; token < valid.length; token++) {
            if (valid[token] != 0 && valid[token] != 1) throw new IllegalArgumentException("Invalid prompt mask");
            bias[token] = valid[token] == 0 ? -10000 : 0;
        }
        return Tensor.from(bias, 1, 1, 1, valid.length);
    }
}
