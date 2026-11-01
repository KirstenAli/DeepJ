package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Random;

/** Pre-normalized, bidirectional encoder block reusing DeepJ linear layers and LayerNorm. */
public final class TransformerEncoderLayer extends Module {
    private final LayerNorm1D attentionNorm;
    private final LayerNorm1D feedForwardNorm;
    private final EncoderAttention attention;
    private final Sequential feedForward;
    private final Dropout attentionDropout;
    private final Dropout outputDropout;

    public TransformerEncoderLayer(TransformerEncoderConfig config, Random random, ParameterFactory parameters) {
        attentionNorm = new LayerNorm1D(config.width(), parameters.scope("attentionNorm"));
        feedForwardNorm = new LayerNorm1D(config.width(), parameters.scope("feedForwardNorm"));
        attention = new EncoderAttention(config.width(), config.heads(), config.dropout(), random,
                parameters.scope("attention"));
        feedForward = new Sequential(
                new Linear(config.width(), config.hiddenSize(), random, parameters.scope("feedForward.input")),
                Activation.gelu(), new Dropout(config.dropout(), random),
                new Linear(config.hiddenSize(), config.width(), random, parameters.scope("feedForward.output")));
        attentionDropout = new Dropout(config.dropout(), random);
        outputDropout = new Dropout(config.dropout(), random);
    }

    @Override protected List<Layer> children() {
        return List.of(attentionNorm, feedForwardNorm, attention, feedForward, attentionDropout, outputDropout);
    }

    @Override public Tensor forward(Tensor input) { return forwardMasked(input, null); }

    public Tensor forward(Tensor input, boolean[][] validTokens) {
        return forwardMasked(input, EncoderMask.create(input, validTokens));
    }

    Tensor forwardMasked(Tensor input, EncoderMask mask) {
        try (var scope = TensorScope.open()) {
            Tensor attended = attention.forwardMasked(attentionNorm.forward(input), mask);
            Tensor residual = input.add(attentionDropout.forward(attended));
            Tensor transformed = feedForward.forward(feedForwardNorm.forward(residual));
            Tensor result = residual.add(outputDropout.forward(transformed));
            return scope.escape(mask == null ? result : mask.apply(result));
        }
    }
}
