package org.deepj.ann.transformer;

import org.deepj.ann.Tensor;
import org.deepj.ann.layers.Layer;
import org.deepj.ann.layers.transformer.TransformerBlock;
import org.deepj.ann.optimisers.Parameter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A simple sequential stack of {@link TransformerBlock}s.
 *
 * <p>This exists to make transformer composition explicit (instead of reusing a generic Sequential).
 */
public final class TransformerStack implements Layer {

    private final List<TransformerBlock> blocks;

    public TransformerStack(List<TransformerBlock> blocks) {
        if (blocks == null) throw new IllegalArgumentException("blocks must not be null");
        if (blocks.isEmpty()) throw new IllegalArgumentException("blocks must be non-empty");
        this.blocks = List.copyOf(blocks);
    }

    public List<TransformerBlock> blocks() {
        return Collections.unmodifiableList(blocks);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor h = x;
        for (TransformerBlock b : blocks) {
            h = b.forward(h);
        }
        return h;
    }

    public Tensor backward(Tensor gradOut) {
        Tensor g = gradOut;
        for (int i = blocks.size() - 1; i >= 0; i--) {
            g = blocks.get(i).backward(g);
        }
        return g;
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

    @Override
    public List<Parameter> parameters() {
        List<Parameter> ps = new ArrayList<>();
        for (TransformerBlock b : blocks) ps.addAll(b.parameters());
        return ps;
    }
}
