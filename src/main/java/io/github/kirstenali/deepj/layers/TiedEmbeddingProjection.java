package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import io.github.kirstenali.deepj.autograd.LayoutOps;

import java.util.List;
import java.util.Objects;

/** Predicts tokens using the same parameter that supplies the input embeddings. */
public final class TiedEmbeddingProjection implements Projection {

    private final Parameter embedding;

    public TiedEmbeddingProjection(Parameter embedding) {
        this.embedding = Objects.requireNonNull(embedding, "embedding");
    }

    @Override
    public Tensor forward(Tensor value) {
        return LayoutOps.batchedMatmul(value, embedding.value, 1, false, true);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(embedding);
    }
}
