package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.ArrayList;
import java.util.List;

abstract class AbstractTransformerBlock implements Layer {

    private final Layer firstNorm;
    private final Layer secondNorm;
    private final Layer attention;
    private final Layer feedForward;

    AbstractTransformerBlock(Layer firstNorm, Layer secondNorm,
                             Layer attention, Layer feedForward) {
        this.firstNorm = firstNorm;
        this.secondNorm = secondNorm;
        this.attention = attention;
        this.feedForward = feedForward;
    }

    static RotaryEmbedding rotaryEmbedding(int dModel, int heads, int maxSequenceLength) {
        if (dModel <= 0 || heads <= 0 || dModel % heads != 0) {
            throw new IllegalArgumentException("dModel must be positive and divisible by nHeads");
        }
        return new RotaryEmbedding(dModel / heads, maxSequenceLength);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor attended = input.add(attention.forward(firstNorm.forward(input)));
        return attended.add(feedForward.forward(secondNorm.forward(attended)));
    }

    @Override
    public Tensor backward(Tensor gradient) {
        Tensor attendedGradient = gradient.add(secondNorm.backward(feedForward.backward(gradient)));
        return attendedGradient.add(firstNorm.backward(attention.backward(attendedGradient)));
    }

    private Layer[] subLayers() {
        return new Layer[]{ firstNorm, secondNorm, attention, feedForward };
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> parameters = new ArrayList<>();
        for (Layer layer : subLayers()) {
            parameters.addAll(layer.parameters());
        }
        return parameters;
    }
}
