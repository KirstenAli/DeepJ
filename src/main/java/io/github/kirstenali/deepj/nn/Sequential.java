package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Arrays;
import java.util.List;

/** Ordered composition of existing or new DeepJ layers. Shared parameters are updated once. */
public final class Sequential extends Module {
    private final List<Layer> layers;

    public Sequential(Layer... layers) { this.layers = List.copyOf(Arrays.asList(layers.clone())); }

    public List<Layer> layers() { return layers; }
    @Override protected List<Layer> children() { return layers; }

    @Override public Tensor forward(Tensor input) {
        try (var scope = TensorScope.open()) {
            Tensor output = input;
            for (Layer layer : layers) { output = layer.forward(output); }
            return scope.escape(output);
        }
    }
}
