package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.SwiGLULayer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Config;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public final class Qwen3Block implements Layer {
    private final RMSNorm1D first, second;
    private final Qwen3Attention attention;
    private final SwiGLULayer feedForward;

    public Qwen3Block(Qwen3Config config, Random random, ParameterFactory parameters) {
        first = new RMSNorm1D(config.dModel(), parameters.scope("input_layernorm"));
        second = new RMSNorm1D(config.dModel(), parameters.scope("post_attention_layernorm"));
        attention = new Qwen3Attention(config, random, parameters.scope("self_attn"));
        feedForward = new SwiGLULayer(config.dModel(), config.dFF(), random, false, parameters.scope("mlp"));
    }

    @Override public Tensor forward(Tensor input) { return finish(input.add(attention.forward(first.forward(input)))); }

    @Override public Tensor forward(Tensor input, InferenceContext context) {
        return finish(input.add(attention.forward(first.forward(input), context)));
    }

    private Tensor finish(Tensor residual) { return residual.add(feedForward.forward(second.forward(residual))); }
    @Override public boolean supportsIncrementalInference() { return true; }
    @Override public List<Parameter> parameters() {
        return Stream.of(first, second, attention, feedForward).flatMap(layer -> layer.parameters().stream()).toList();
    }
}
