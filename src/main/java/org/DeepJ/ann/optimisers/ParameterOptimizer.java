package org.DeepJ.ann.optimisers;

import java.util.List;

/**
 * Optimizer that updates a set of Parameters once per training step.
 *
 * <p>Important: implementations should treat one call to {@link #step(List)}
 * as a single optimizer step (for bias correction, schedules, etc.).
 */
public interface ParameterOptimizer {

    /** Update the provided parameters (one optimizer step). */
    void step(List<Parameter> params);
}