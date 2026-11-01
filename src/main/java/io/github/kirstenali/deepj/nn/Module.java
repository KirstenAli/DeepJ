package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/** Additive layer base with recursive training mode and identity-deduplicated parameters. */
public abstract class Module implements Layer {
    private boolean training = true;

    public final boolean isTraining() { return training; }
    public final void train() { setTraining(true); }
    public final void eval() { setTraining(false); }

    public final void setTraining(boolean enabled) {
        training = enabled;
        for (Layer child : children()) {
            if (child instanceof Module module) module.setTraining(enabled);
        }
    }

    protected List<Layer> children() { return List.of(); }

    @Override public List<Parameter> parameters() {
        var seen = new IdentityHashMap<Tensor, Boolean>();
        List<Parameter> result = new ArrayList<>();
        for (Layer child : children()) {
            for (Parameter parameter : child.parameters()) {
                if (seen.put(parameter.value, Boolean.TRUE) == null) result.add(parameter);
            }
        }
        return List.copyOf(result);
    }

    /** Persistent non-parameter tensors, in deterministic traversal order. */
    public List<Tensor> buffers() {
        List<Tensor> result = new ArrayList<>();
        for (Layer child : children()) {
            if (child instanceof Module module) result.addAll(module.buffers());
        }
        return List.copyOf(result);
    }
}
