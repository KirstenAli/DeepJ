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
    private final float residualScale;

    AbstractTransformerBlock(Layer firstNorm, Layer secondNorm,
                             Layer attention, Layer feedForward) {
        this(firstNorm, secondNorm, attention, feedForward, 1.0f);
    }

    AbstractTransformerBlock(Layer firstNorm, Layer secondNorm,
                             Layer attention, Layer feedForward, float residualScale) {
        if (!Float.isFinite(residualScale) || residualScale <= 0.0f) {
            throw new IllegalArgumentException("residualScale must be finite and > 0");
        }
        this.firstNorm = firstNorm;
        this.secondNorm = secondNorm;
        this.attention = attention;
        this.feedForward = feedForward;
        this.residualScale = residualScale;
    }

    static RotaryEmbedding rotaryEmbedding(int dModel, int heads, int maxSequenceLength) {
        if (dModel <= 0 || heads <= 0 || dModel % heads != 0) {
            throw new IllegalArgumentException("dModel must be positive and divisible by nHeads");
        }
        return new RotaryEmbedding(dModel / heads, maxSequenceLength);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor attended = residual(input, firstNorm, attention);
        return residual(attended, secondNorm, feedForward);
    }

    private Tensor residual(Tensor input, Layer norm, Layer branch) {
        Tensor update = branch.forward(norm.forward(input));
        return input.add(residualScale == 1.0f ? update : update.multiplyScalar(residualScale));
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
