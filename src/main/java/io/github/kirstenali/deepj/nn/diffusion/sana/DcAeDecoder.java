package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** The DC-AE decoder used by SANA, including its multiscale linear attention and spatial shortcuts. */
public final class DcAeDecoder {
    private final SanaConfig config;
    private final SanaWeights weights;
    private final int[] channels, layers;

    public DcAeDecoder(Path config, SanaWeights weights) throws IOException {
        this.config = new SanaConfig(config);
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
        try (var scope = TensorScope.open(); var pinned = SanaMemory.retain(latent)) {
            Tensor hidden = SanaConvolution.apply(latent, weights, "decoder.conv_in")
                    .add(SanaOps.repeat(latent, 1, channels[channels.length - 1] / latent.size(1)));
            for (int stage = channels.length - 1; stage >= 0; stage--) {
                hidden = stage(hidden, stage);
            }
            hidden = SanaConvolution.channelNorm(hidden, weights, "decoder.norm_out").reluActivation();
            return scope.escape(SanaConvolution.apply(hidden, weights, "decoder.conv_out"));
        }
    }

    private Tensor stage(Tensor input, int stage) {
        try (var scope = TensorScope.open()) {
            String prefix = "decoder.up_blocks." + stage + ".";
            int offset = stage < channels.length - 1 ? 1 : 0;
            Tensor hidden = offset == 0 ? input : upsample(input, prefix + "0");
            for (int layer = 0; layer < layers[stage]; layer++) {
                SanaMemory.checkCancelled();
                hidden = block(hidden, stage, prefix + (layer + offset));
            }
            return scope.escape(SanaMemory.finish(hidden));
        }
    }

    private Tensor block(Tensor input, int stage, String name) {
        String kind = (String) config.array("decoder_block_types").get(stage);
        try (var scope = TensorScope.open()) {
            Tensor output = switch (kind) {
                case "ResBlock" -> residual(input, name);
                case "EfficientViTBlock" -> SanaConvolution.gated(multiscale(input, stage, name + ".attn"),
                        weights, name + ".conv_out", true);
                default -> throw new IllegalArgumentException("Unsupported DC-AE decoder block: " + kind);
            };
            return scope.escape(SanaMemory.finish(output));
        }
    }

    private Tensor residual(Tensor input, String name) {
        Tensor hidden = SanaOps.silu(SanaConvolution.apply(input, weights, name + ".conv1"));
        hidden = SanaConvolution.apply(hidden, weights, name + ".conv2");
        return SanaConvolution.channelNorm(hidden, weights, name + ".norm").add(input);
    }

    private Tensor upsample(Tensor input, String name) {
        Tensor result = SanaConvolution.apply(SanaOps.upsample(input), weights, name + ".conv");
        int repeats = result.size(1) * 4 / input.size(1);
        return result.add(SanaOps.pixelShuffle(SanaOps.repeat(input, 1, repeats)));
    }

    private Tensor multiscale(Tensor input, int stage, String name) {
        Tensor tokens = input.permute(0, 2, 3, 1);
        Tensor query = SanaOps.linear(tokens, weights, name + ".to_q");
        Tensor key = SanaOps.linear(tokens, weights, name + ".to_k");
        Tensor value = SanaOps.linear(tokens, weights, name + ".to_v");
        Tensor base = SanaOps.cat(3, query, key, value).permute(0, 3, 1, 2);
        List<Tensor> features = new ArrayList<>(List.of(base));
        var scales = (List<?>) config.array("decoder_qkv_multiscales").get(stage);
        for (int index = 0; index < scales.size(); index++) {
            String prefix = name + ".to_qkv_multiscale." + index;
            Tensor projection = SanaConvolution.apply(base, weights, prefix + ".proj_in");
            features.add(SanaConvolution.apply(projection, weights, prefix + ".proj_out"));
        }
        Tensor result = attend(SanaOps.cat(1, features.toArray(Tensor[]::new)), input);
        result = SanaOps.linear(result.permute(0, 2, 3, 1), weights, name + ".to_out").permute(0, 3, 1, 2);
        return SanaConvolution.channelNorm(result, weights, name + ".norm_out").add(input);
    }

    private Tensor attend(Tensor features, Tensor input) {
        int dim = config.integer("attention_head_dim"), positions = input.size(2) * input.size(3);
        Tensor grouped = features.reshape(input.size(0), features.size(1) / (3 * dim), 3 * dim, positions);
        Tensor query = grouped.slice(2, 0, dim).transpose(-1, -2);
        Tensor key = grouped.slice(2, dim, dim * 2).transpose(-1, -2);
        Tensor value = grouped.slice(2, dim * 2, dim * 3).transpose(-1, -2);
        return SanaOps.linearAttention(query, key, value, 1e-15f).transpose(-1, -2)
                .reshape(input.size(0), -1, input.size(2), input.size(3));
    }
}
