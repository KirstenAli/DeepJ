package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.Collections;
import java.util.List;

public interface Trainable {

    default List<Parameter> parameters() {
        return Collections.emptyList();
    }

    default void zeroGrad() {
        for (Parameter p : parameters()) p.zeroGrad();
    }
}
