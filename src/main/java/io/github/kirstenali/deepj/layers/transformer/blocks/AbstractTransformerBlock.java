package io.github.kirstenali.deepj.layers.transformer.blocks;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for transformer blocks.
 *
 * <p>Subclasses declare their sub-layers as fields and expose them via
 * {@link #subLayers()}; {@link #parameters()} is implemented here once.
 */
abstract class AbstractTransformerBlock implements Layer {

    /** Return the ordered sub-layers that own trainable parameters. */
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
