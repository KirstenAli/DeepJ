package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;

/** Max/average pooling of [batch, channels, length] inputs. */
public final class Pool1D extends Module {
    private final Pool2D pooling;

    public Pool1D(int kernel, int stride, int padding, Pool2D.Reduction reduction) {
        pooling = new Pool2D(new Window2D(1, kernel, 1, stride, 0, padding, 1, 1), reduction);
    }

    public static Pool1D max(int kernel) { return new Pool1D(kernel, kernel, 0, Pool2D.Reduction.MAX); }
    public static Pool1D average(int kernel) { return new Pool1D(kernel, kernel, 0, Pool2D.Reduction.AVERAGE); }
    @Override protected List<Layer> children() { return List.of(pooling); }

    @Override public Tensor forward(Tensor input) {
        NeuralChecks.rank(input, 3);
        try (var scope = TensorScope.open()) { return scope.escape(pooling.forward(input.unsqueeze(2)).select(2, 0)); }
    }
}
