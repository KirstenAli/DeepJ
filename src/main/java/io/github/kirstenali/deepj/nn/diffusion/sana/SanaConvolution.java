package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.nn.GroupedConv2D;
import io.github.kirstenali.deepj.tensor.Tensor;

final class SanaConvolution {
    private SanaConvolution() {}

    static Tensor apply(Tensor input, SanaWeights weights, String name) {
        Tensor weight = weights.tensor(name + ".weight");
        int groups = input.size(1) / weight.size(1);
        Tensor bias = weights.contains(name + ".bias") ? weights.tensor(name + ".bias") : null;
        return GroupedConv2D.apply(input, weight, bias, groups);
    }

    static Tensor gated(Tensor input, SanaWeights weights, String name, boolean residual) {
        Tensor hidden = SanaOps.silu(apply(input, weights, name + ".conv_inverted"));
        hidden = apply(hidden, weights, name + ".conv_depth");
        int width = hidden.size(1) / 2;
        hidden = hidden.slice(1, 0, width).multiply(SanaOps.silu(hidden.slice(1, width, width * 2)));
        hidden = apply(hidden, weights, name + ".conv_point");
        if (weights.contains(name + ".norm.weight")) hidden = channelNorm(hidden, weights, name + ".norm");
        return residual ? hidden.add(input) : hidden;
    }

    static Tensor channelNorm(Tensor input, SanaWeights weights, String name) {
        Tensor normalized = SanaOps.affineRms(input.permute(0, 2, 3, 1), weights, name, 1e-5f, false);
        return normalized.permute(0, 3, 1, 2);
    }
}
