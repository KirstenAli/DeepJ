package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.activations.Softmax;
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
    private final GroupedHeads groups;
    private final Softmax softmax = new Softmax();
    private Cache cache;

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
        groups = new GroupedHeads(config.nHeads(), config.kvHeads());
    }

    @Override
    public Tensor forward(Tensor input) {
        int length = input.rows;
        Tensor queries = transform(queryNorm, query.forward(input), length, config.nHeads());
        Tensor keys = transform(keyNorm, key.forward(input), length, config.kvHeads());
        Tensor values = HeadOps.splitHeads(value.forward(input), config.kvHeads());
        return attend(queries, groups.expand(keys, length), groups.expand(values, length), length);
    }

    private Tensor transform(HeadNormalization norm, Tensor projection, int length, int heads) {
        return rotary.apply(norm.forward(HeadOps.splitHeads(projection, heads)), length, heads);
    }

    private Tensor attend(Tensor queries, Tensor keys, Tensor values, int length) {
        Tensor scores = HeadOps.dotProductScores(queries, keys, config.nHeads());
        Tensor probabilities = softmax.forwardCausal(scores, length, scale());
        Tensor context = HeadOps.applyAttentionToValues(probabilities, values, config.nHeads());
        cache = new Cache(queries, keys, values, probabilities, length);
        return output.forward(HeadOps.mergeHeads(context, config.nHeads()));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor heads = HeadOps.splitHeads(output.backward(gradient), config.nHeads());
        var attention = HeadOps.backwardAttentionAndValues(heads, cache.values(),
                cache.probabilities(), softmax, scale(), config.nHeads());
        var projections = HeadOps.backwardQueriesAndKeys(attention.dScores(),
                cache.queries(), cache.keys(), config.nHeads());
        return backwardProjections(projections, attention.dVh());
    }

    private Tensor backwardProjections(HeadOps.QKGrads gradients, Tensor valueGradient) {
        Tensor queryGradient = reverseTransform(queryNorm, gradients.dQh(), config.nHeads());
        Tensor keys = groups.reduce(gradients.dKh(), cache.length());
        Tensor keyGradient = reverseTransform(keyNorm, keys, config.kvHeads());
        Tensor values = groups.reduce(valueGradient, cache.length());
        return query.backward(queryGradient).add(key.backward(keyGradient))
                .add(value.backward(HeadOps.mergeHeads(values, config.kvHeads())));
    }

    private Tensor reverseTransform(HeadNormalization norm, Tensor gradient, int heads) {
        Tensor rotated = rotary.applyBackward(gradient, cache.length(), heads);
        return HeadOps.mergeHeads(norm.backward(rotated), heads);
    }

    private float scale() {
        return (float) (1.0 / Math.sqrt(config.dModel() / config.nHeads()));
    }

    @Override
    public List<Parameter> parameters() {
        return Stream.of(query, key, value, output, queryNorm, keyNorm)
                .flatMap(layer -> layer.parameters().stream()).toList();
    }

    private record Cache(Tensor queries, Tensor keys, Tensor values, Tensor probabilities, int length) {}
}
