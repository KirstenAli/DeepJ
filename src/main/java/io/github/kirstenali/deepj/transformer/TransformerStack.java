package io.github.kirstenali.deepj.transformer;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.blocks.DeepSeekTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.GPTTransformerBlock;
import io.github.kirstenali.deepj.layers.transformer.blocks.LlamaTransformerBlock;
import io.github.kirstenali.deepj.optimisers.Parameter;

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
    public Tensor backward(Tensor gradOut) {
        Tensor g = gradOut;
        for (int i = blocks.size() - 1; i >= 0; i--) {
            g = blocks.get(i).backward(g);
        }
        return g;
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (Layer b : blocks) {
            ps.addAll(b.parameters());
        }
        return ps;
    }
}
