package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

/** One generation request. Cached sessions return only the last logit row. Close to release caches. */
@FunctionalInterface
public interface GenerationSession extends Function<int[], Tensor>, AutoCloseable {
    @Override
    default void close() {}
}
