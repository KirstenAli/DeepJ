package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.inference.InferenceContext;
import io.github.kirstenali.deepj.layers.BiasFreeLinear;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

public final class GroupedQueryAttention implements Layer {

    private final DeepJNovaConfig config;
    private final BiasFreeLinear query, key, value, output;
    private final HeadNormalization queryNorm, keyNorm;
    private final RotaryEmbedding rotary;

    public GroupedQueryAttention(DeepJNovaConfig config, Random random) {
        this.config = config;
        int width = config.dModel(), headWidth = width / config.nHeads();
        int keyValueWidth = config.kvHeads() * headWidth;
        query = new BiasFreeLinear(width, width, random);
        key = new BiasFreeLinear(width, keyValueWidth, random);
        value = new BiasFreeLinear(width, keyValueWidth, random);
        output = new BiasFreeLinear(width, width, random);
        queryNorm = new HeadNormalization(headWidth, config.qkNorm());
        keyNorm = new HeadNormalization(headWidth, config.qkNorm());
        rotary = new RotaryEmbedding(headWidth, config.maxSeqLen());
    }

    @Override
    public Tensor forward(Tensor input) {
        int length = input.rows;
        Tensor queries = transform(queryNorm, query.forward(input), length, config.nHeads());
        Tensor keys = transform(keyNorm, key.forward(input), length, config.kvHeads());
        Tensor values = HeadOps.splitHeads(value.forward(input), config.kvHeads());
        return attend(queries, keys, values);
    }

    private Tensor transform(HeadNormalization norm, Tensor projection, int length, int heads) {
        return rotary.apply(norm.forward(HeadOps.splitHeads(projection, heads)), length, heads);
    }

    private Tensor attend(Tensor queries, Tensor keys, Tensor values) {
        Tensor context = AttentionOps.causal(
                queries, keys, values, config.nHeads(), config.kvHeads(), scale());
        return output.forward(HeadOps.mergeHeads(context, config.nHeads()));
    }

    private float scale() {
        return (float) (1.0 / Math.sqrt(config.dModel() / config.nHeads()));
    }

    @Override
    public boolean supportsIncrementalInference() { return true; }

    @Override
    public Tensor forward(Tensor input, InferenceContext context) {
        Tensor queries = cachedTransform(queryNorm, query.forward(input), config.nHeads(), context.position());
        Tensor keys = cachedTransform(keyNorm, key.forward(input), config.kvHeads(), context.position());
        Tensor values = HeadOps.splitHeads(value.forward(input), config.kvHeads());
        Tensor attended = context.attend(this, queries, keys, values, config.nHeads(), config.kvHeads(), scale());
        return output.forward(HeadOps.mergeHeads(attended, config.nHeads()));
    }

    private Tensor cachedTransform(HeadNormalization norm, Tensor projection, int heads, int offset) {
        Tensor split = norm.forward(HeadOps.splitHeads(projection, heads));
        return rotary.apply(split, projection.rows, heads, offset);
    }

    @Override
    public List<Parameter> parameters() {
        return Stream.of(query, key, value, output, queryNorm, keyNorm)
                .flatMap(layer -> layer.parameters().stream()).toList();
    }

}
