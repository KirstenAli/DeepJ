package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Objects;

/** Explicit caller-owned state. Cell memory is present only for LSTM. */
public record RecurrentState(Tensor hidden, Tensor cell) {
    public RecurrentState { Objects.requireNonNull(hidden, "hidden"); }
    public RecurrentState(Tensor hidden) { this(hidden, null); }
    public RecurrentState detach() { return new RecurrentState(hidden.detach(), cell == null ? null : cell.detach()); }
}
