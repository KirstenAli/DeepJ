package io.github.kirstenali.deepj.layers.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.autograd.LayoutOps;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Projection;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3Config;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public final class Qwen3Attention implements Layer {
    private final Qwen3Config config;
    private final Projection query, key, value, output;
    private final RMSNorm1D queryNorm, keyNorm;
    private final Qwen3Rotary rotary;

    public Qwen3Attention(Qwen3Config config, Random random, ParameterFactory parameters) {
        this.config = config;
        int queries = config.nHeads() * config.headDim(), keys = config.kvHeads() * config.headDim();
        query = new BiasFreeLinear(config.dModel(), queries, random, parameters.scope("q_proj"));
        key = new BiasFreeLinear(config.dModel(), keys, random, parameters.scope("k_proj"));
        value = new BiasFreeLinear(config.dModel(), keys, random, parameters.scope("v_proj"));
        output = new BiasFreeLinear(queries, config.dModel(), random, parameters.scope("o_proj"));
        queryNorm = new RMSNorm1D(config.headDim(), parameters.scope("q_norm"));
        keyNorm = new RMSNorm1D(config.headDim(), parameters.scope("k_norm"));
        rotary = new Qwen3Rotary(config.headDim(), config.ropeTheta());
    }

    Qwen3Attention(Qwen3Config config, ParameterFactory parameters,
                   java.util.function.Function<String, Projection> projection) {
        this.config = config;
        query = projection.apply("q_proj");
        key = projection.apply("k_proj");
        value = projection.apply("v_proj");
        output = projection.apply("o_proj");
        queryNorm = new RMSNorm1D(config.headDim(), parameters.scope("q_norm"));
        keyNorm = new RMSNorm1D(config.headDim(), parameters.scope("k_norm"));
        rotary = new Qwen3Rotary(config.headDim(), config.ropeTheta());
    }

    @Override public Tensor forward(Tensor input) { return attend(input, null); }
    @Override public Tensor forward(Tensor input, InferenceContext context) { return attend(input, context); }
    @Override public boolean supportsIncrementalInference() { return true; }

    private Tensor attend(Tensor input, InferenceContext context) {
        int offset = context == null ? 0 : context.position(), length = input.size(-2);
        Tensor q = rotated(queryNorm, query.forward(input), config.nHeads(), length, offset);
        Tensor k = rotated(keyNorm, key.forward(input), config.kvHeads(), length, offset);
        Tensor v = LayoutOps.splitHeads(value.forward(input), config.kvHeads());
        float scale = (float) (1 / Math.sqrt(config.headDim()));
        Tensor result = context == null ? AttentionOps.causal(q, k, v, config.nHeads(), config.kvHeads(), scale)
                : context.attend(this, q, k, v, config.nHeads(), config.kvHeads(), scale);
        return output.forward(LayoutOps.mergeHeads(result, config.nHeads()));
    }

    private Tensor rotated(RMSNorm1D norm, Tensor input, int heads, int length, int offset) {
        return rotary.apply(norm.forward(LayoutOps.splitHeads(input, heads)), length, offset);
    }

    @Override public List<Parameter> parameters() {
        return Stream.of(query, key, value, output, queryNorm, keyNorm)
                .flatMap(layer -> layer.parameters().stream()).toList();
    }
}
