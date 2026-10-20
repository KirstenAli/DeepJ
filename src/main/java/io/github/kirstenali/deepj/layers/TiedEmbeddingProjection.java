package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Objects;

/** Predicts tokens using the same parameter that supplies the input embeddings. */
public final class TiedEmbeddingProjection implements Projection {

    private final Parameter embedding;
    private Tensor input;

    public TiedEmbeddingProjection(Parameter embedding) {
        this.embedding = Objects.requireNonNull(embedding, "embedding");
    }

    @Override
    public Tensor forward(Tensor value) {
        input = value;
        return Tensor.backend().batchedMatmul(value, embedding.value, 1, false, true);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        embedding.grad.addInPlace(Tensor.backend().batchedMatmul(gradient, input, 1, true, false));
        return gradient.matmul(embedding.value);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(embedding);
    }
}
