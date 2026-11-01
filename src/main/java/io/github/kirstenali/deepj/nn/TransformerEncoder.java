package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.norm.LayerNorm1D;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Stacked encoder for [batch, time, width] embeddings, with padding masks and masked mean pooling. */
public final class TransformerEncoder extends Module {
    private final TransformerEncoderConfig config;
    private final List<TransformerEncoderLayer> blocks;
    private final LayerNorm1D finalNorm;

    public TransformerEncoder(TransformerEncoderConfig config, Random random) {
        this(config, random, ParameterFactory.float32());
    }

    public TransformerEncoder(TransformerEncoderConfig config, Random random, ParameterFactory parameters) {
        this.config = config;
        List<TransformerEncoderLayer> layers = new ArrayList<>();
        for (int index = 0; index < config.layers(); index++) {
            layers.add(new TransformerEncoderLayer(config, random, parameters.scope("blocks.layer" + index)));
        }
        blocks = List.copyOf(layers);
        finalNorm = new LayerNorm1D(config.width(), parameters.scope("finalNorm"));
    }

    public TransformerEncoderConfig config() { return config; }

    @Override protected List<Layer> children() {
        List<Layer> result = new ArrayList<>(blocks);
        result.add(finalNorm);
        return List.copyOf(result);
    }

    @Override public Tensor forward(Tensor input) { return forward(input, (boolean[][]) null); }

    public Tensor forward(Tensor input, boolean[][] validTokens) {
        try (var scope = TensorScope.open()) {
            return scope.escape(encode(input, EncoderMask.create(input, validTokens)));
        }
    }

    public Tensor meanPool(Tensor input, boolean[][] validTokens) {
        try (var scope = TensorScope.open()) {
            EncoderMask mask = EncoderMask.create(input, validTokens);
            Tensor output = encode(input, mask);
            return scope.escape(mask == null ? output.mean(1, false) : output.sum(1, false).divide(mask.counts()));
        }
    }

    private Tensor encode(Tensor input, EncoderMask mask) {
        NeuralChecks.width(input, config.width());
        Tensor result = input;
        for (TransformerEncoderLayer block : blocks) { result = block.forwardMasked(result, mask); }
        result = finalNorm.forward(result);
        return mask == null ? result : mask.apply(result);
    }
}
