package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.stream.IntStream;

/** Averages every spatial axis of [batch, channels, ...], returning [batch, channels]. */
public final class GlobalAveragePool extends Module {
    @Override public Tensor forward(Tensor input) {
        if (input.rank() < 3 || input.numel() == 0) {
            throw new IllegalArgumentException("Global pooling needs nonempty batch, channel and spatial axes");
        }
        return input.meanAxes(IntStream.range(2, input.rank()).toArray());
    }
}
