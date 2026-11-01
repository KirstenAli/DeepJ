package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.Objects;

/** Floor-mode max/average pooling of channel-first images. Average pooling includes padded zeros. */
public final class Pool2D extends Module {
    public enum Reduction { MAX, AVERAGE }
    private final Window2D window;
    private final Reduction reduction;

    public Pool2D(Window2D window, Reduction reduction) {
        this.window = Objects.requireNonNull(window, "window");
        this.reduction = Objects.requireNonNull(reduction, "reduction");
    }

    public static Pool2D max(int kernel) { return new Pool2D(downsample(kernel), Reduction.MAX); }
    public static Pool2D average(int kernel) { return new Pool2D(downsample(kernel), Reduction.AVERAGE); }

    private static Window2D downsample(int kernel) {
        return new Window2D(kernel, kernel, kernel, kernel, 0, 0, 1, 1);
    }

    @Override public Tensor forward(Tensor input) {
        try (var scope = TensorScope.open()) {
            float padding = reduction == Reduction.MAX ? Float.NEGATIVE_INFINITY : 0;
            SpatialPatches patches = SpatialPatches.gather(input, window, padding);
            Tensor result = reduction == Reduction.MAX ? patches.values().maximum(-1, false)
                    : patches.values().mean(-1, false);
            return scope.escape(result.reshape(input.size(0), input.size(1), patches.height(), patches.width()));
        }
    }
}
