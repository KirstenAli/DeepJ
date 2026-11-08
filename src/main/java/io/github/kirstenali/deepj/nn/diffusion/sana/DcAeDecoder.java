package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import io.github.kirstenali.deepj.nn.ModelOps;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.ModelInference;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** The DC-AE decoder used by SANA, including its multiscale linear attention and spatial shortcuts. */
public final class DcAeDecoder {
    private final HuggingFaceConfig config;
    private final TensorWeights weights;
    private final int[] channels, layers;

    public DcAeDecoder(Path config, TensorWeights weights) throws IOException {
        this.config = new HuggingFaceConfig(config);
        this.config.require("_class_name", "AutoencoderDC");
        this.config.require("upsample_block_type", "interpolate");
        channels = this.config.integers("decoder_block_out_channels");
        layers = this.config.integers("decoder_layers_per_block");
        if (channels.length != layers.length) throw new IllegalArgumentException("Invalid DC-AE stage counts");
        this.weights = weights;
    }

    public Tensor forward(Tensor latent) {
        if (latent.rank() != 4 || latent.size(1) != config.integer("latent_channels")) {
            throw new IllegalArgumentException("Invalid DC-AE latent channels");
        }
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(latent)) {
            Tensor hidden = SanaConvolution.apply(latent, weights, "decoder.conv_in")
                    .add(ModelOps.repeat(latent, 1, channels[channels.length - 1] / latent.size(1)));
            for (int stage = channels.length - 1; stage >= 0; stage--) {
                hidden = stage(hidden, stage);
            }
            return scope.escape(output(hidden));
        }
    }

    private Tensor output(Tensor input) {
        return ModelInference.evaluate(() -> {
            Tensor hidden = SanaConvolution.channelNorm(input, weights, "decoder.norm_out").reluActivation();
            return SanaConvolution.apply(hidden, weights, "decoder.conv_out");
        });
    }

    private Tensor stage(Tensor input, int stage) {
        try (var scope = TensorScope.open()) {
            String prefix = "decoder.up_blocks." + stage + ".";
            int offset = stage < channels.length - 1 ? 1 : 0;
            Tensor hidden = offset == 0 ? input : upsample(input, prefix + "0");
            for (int layer = 0; layer < layers[stage]; layer++) {
                ModelMemory.checkCancelled();
                hidden = block(hidden, stage, prefix + (layer + offset));
            }
            return scope.escape(hidden);
        }
    }

    private Tensor block(Tensor input, int stage, String name) {
        return ModelInference.evaluate(() -> blockValues(input, stage, name));
    }

    private Tensor blockValues(Tensor input, int stage, String name) {
        String kind = (String) config.array("decoder_block_types").get(stage);
        try (var scope = TensorScope.open()) {
            Tensor output = switch (kind) {
                case "ResBlock" -> residual(input, name);
                case "EfficientViTBlock" -> SanaConvolution.gated(multiscale(input, stage, name + ".attn"),
                        weights, name + ".conv_out", true);
                default -> throw new IllegalArgumentException("Unsupported DC-AE decoder block: " + kind);
            };
            return scope.escape(output);
        }
    }

    private Tensor residual(Tensor input, String name) {
        Tensor hidden = ModelOps.silu(SanaConvolution.apply(input, weights, name + ".conv1"));
        hidden = SanaConvolution.apply(hidden, weights, name + ".conv2");
        return SanaConvolution.channelNorm(hidden, weights, name + ".norm").add(input);
    }

    private Tensor upsample(Tensor input, String name) {
        return ModelInference.evaluate(() -> upsampleValues(input, name));
    }

    private Tensor upsampleValues(Tensor input, String name) {
        Tensor result = SanaConvolution.apply(ModelOps.upsample(input), weights, name + ".conv");
        int repeats = result.size(1) * 4 / input.size(1);
        return result.add(ModelOps.pixelShuffle(ModelOps.repeat(input, 1, repeats)));
    }

    private Tensor multiscale(Tensor input, int stage, String name) {
        Tensor tokens = input.permute(0, 2, 3, 1);
        Tensor query = ModelOps.linear(tokens, weights, name + ".to_q");
        Tensor key = ModelOps.linear(tokens, weights, name + ".to_k");
        Tensor value = ModelOps.linear(tokens, weights, name + ".to_v");
        Tensor base = ModelOps.cat(3, query, key, value).permute(0, 3, 1, 2);
        List<Tensor> features = new ArrayList<>(List.of(base));
        var scales = (List<?>) config.array("decoder_qkv_multiscales").get(stage);
        for (int index = 0; index < scales.size(); index++) {
            String prefix = name + ".to_qkv_multiscale." + index;
            Tensor projection = SanaConvolution.apply(base, weights, prefix + ".proj_in");
            features.add(SanaConvolution.apply(projection, weights, prefix + ".proj_out"));
        }
        Tensor result = attend(ModelOps.cat(1, features.toArray(Tensor[]::new)), input);
        result = ModelOps.linear(result.permute(0, 2, 3, 1), weights, name + ".to_out").permute(0, 3, 1, 2);
        return SanaConvolution.channelNorm(result, weights, name + ".norm_out").add(input);
    }

    private Tensor attend(Tensor features, Tensor input) {
        int dim = config.integer("attention_head_dim"), positions = input.size(2) * input.size(3);
        Tensor grouped = features.reshape(input.size(0), features.size(1) / (3 * dim), 3 * dim, positions);
        Tensor query = grouped.slice(2, 0, dim).transpose(-1, -2);
        Tensor key = grouped.slice(2, dim, dim * 2).transpose(-1, -2);
        Tensor value = grouped.slice(2, dim * 2, dim * 3).transpose(-1, -2);
        return ModelOps.linearAttention(query, key, value, 1e-15f).transpose(-1, -2)
                .reshape(input.size(0), -1, input.size(2), input.size(3));
    }
}
