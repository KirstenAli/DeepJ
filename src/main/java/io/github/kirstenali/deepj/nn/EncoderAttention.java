package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Random;

/** Bidirectional multi-head attention with optional padding masks and backend-resident softmax. */
public final class EncoderAttention extends Module {
    private final int width;
    private final int heads;
    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Linear output;
    private final Dropout dropout;

    public EncoderAttention(int width, int heads, float dropout, Random random, ParameterFactory parameters) {
        this.width = NeuralChecks.positive(width, "model width");
        this.heads = NeuralChecks.positive(heads, "heads");
        if (width % heads != 0) throw new IllegalArgumentException("Model width must be divisible by heads");
        query = new Linear(width, width, random, parameters.scope("query"));
        key = new Linear(width, width, random, parameters.scope("key"));
        value = new Linear(width, width, random, parameters.scope("value"));
        output = new Linear(width, width, random, parameters.scope("output"));
        this.dropout = new Dropout(dropout, random);
    }

    @Override protected List<Layer> children() { return List.of(query, key, value, output, dropout); }
    @Override public Tensor forward(Tensor input) { return forwardMasked(input, null); }

    public Tensor forward(Tensor input, boolean[][] validTokens) {
        return forwardMasked(input, EncoderMask.create(input, validTokens));
    }

    Tensor forwardMasked(Tensor input, EncoderMask mask) {
        NeuralChecks.rank(input, 3);
        NeuralChecks.width(input, width);
        try (var scope = TensorScope.open()) {
            Tensor q = split(query.forward(input)), k = split(key.forward(input)), v = split(value.forward(input));
            Tensor scores = q.matmul(k.transpose(-2, -1)).multiplyScalar((float) (1 / Math.sqrt(width / heads)));
            if (mask != null) scores = scores.add(mask.keys());
            Tensor probabilities = dropout.forward(scores.softmax(-1));
            Tensor context = probabilities.matmul(v).permute(0, 2, 1, 3).reshape(input.shape());
            Tensor result = output.forward(context);
            return scope.escape(mask == null ? result : mask.apply(result));
        }
    }

    private Tensor split(Tensor projection) {
        return projection.to(DType.FLOAT32).reshape(projection.size(0), projection.size(1), heads, width / heads)
                .permute(0, 2, 1, 3);
    }
}
