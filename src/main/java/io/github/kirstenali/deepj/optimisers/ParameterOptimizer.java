package io.github.kirstenali.deepj.optimisers;

import java.util.List;

public interface ParameterOptimizer {

    void step(List<Parameter> params);
}
