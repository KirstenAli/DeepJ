package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.inference.InferenceContext;

import java.util.ArrayList;
import java.util.List;

public record TransformerStack(List<Layer> blocks) implements Layer {

    public TransformerStack(List<Layer> blocks) {
        if (blocks == null) throw new IllegalArgumentException("blocks must not be null");
        if (blocks.isEmpty()) throw new IllegalArgumentException("blocks must be non-empty");
        this.blocks = List.copyOf(blocks);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor h = x;
        for (Layer b : blocks) {
            h = b.forward(h);
        }
        return h;
    }

    @Override
    public Tensor forward(Tensor input, InferenceContext context) {
        Tensor hidden = input;
        for (Layer block : blocks) {
            hidden = block.forward(hidden, context);
        }
        return hidden;
    }

    @Override
    public boolean supportsIncrementalInference() {
        return blocks.stream().allMatch(Layer::supportsIncrementalInference);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (Layer b : blocks) {
            ps.addAll(b.parameters());
        }
        return ps.stream().distinct().toList();
    }
}
