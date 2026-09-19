package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;

abstract class AbstractTransformerBlock implements Layer {

    protected abstract Layer[] subLayers();

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (Layer l : subLayers()) {
            ps.addAll(l.parameters());
        }
        return ps;
    }
}
