package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.Linear;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Random;

/** Shared input/state validation and DeepJ linear projections for recurrent cells. */
public abstract class RecurrentCell extends Module {
    private final int inputSize;
    private final int hiddenSize;
    protected final Linear inputProjection;
    protected final Linear hiddenProjection;

    protected RecurrentCell(int inputs, int hidden, int gates, Random random, ParameterFactory parameters) {
        inputSize = NeuralChecks.positive(inputs, "input size");
        hiddenSize = NeuralChecks.positive(hidden, "hidden size");
        int width = Math.multiplyExact(hidden, gates);
        inputProjection = new Linear(inputs, width, random, parameters.scope("input"));
        hiddenProjection = new Linear(hidden, width, random, parameters.scope("hidden"));
    }

    public final int inputSize() { return inputSize; }
    public final int hiddenSize() { return hiddenSize; }
    protected boolean hasCellMemory() { return false; }
    protected abstract RecurrentState next(Tensor input, RecurrentState state);
    @Override protected List<Layer> children() { return List.of(inputProjection, hiddenProjection); }

    public final RecurrentState initialState(int batch, DType dtype) {
        NeuralChecks.positive(batch, "batch size");
        Tensor hidden = Tensor.backend().zeros(dtype, batch, hiddenSize);
        Tensor memory = hasCellMemory() ? Tensor.backend().zeros(dtype, batch, hiddenSize) : null;
        return new RecurrentState(hidden, memory);
    }

    public final RecurrentState step(Tensor input, RecurrentState state) {
        validate(input, state);
        try (var scope = TensorScope.open()) {
            RecurrentState result = next(input, state);
            scope.escape(result.hidden());
            if (result.cell() != null) scope.escape(result.cell());
            return result;
        }
    }

    @Override public final Tensor forward(Tensor input) {
        NeuralChecks.rank(input, 2);
        return step(input, initialState(input.size(0), input.dtype())).hidden();
    }

    private void validate(Tensor input, RecurrentState state) {
        NeuralChecks.rank(input, 2);
        NeuralChecks.width(input, inputSize);
        if (state == null) throw new IllegalArgumentException("Recurrent state is required");
        requireState(input, state.hidden());
        if (hasCellMemory()) requireState(input, state.cell());
        else if (state.cell() != null) throw new IllegalArgumentException("This cell has no cell memory");
    }

    private void requireState(Tensor input, Tensor state) {
        NeuralChecks.rank(state, 2);
        if (state.size(0) != input.size(0) || state.size(1) != hiddenSize) {
            throw new IllegalArgumentException("Recurrent state must match batch and hidden size");
        }
    }

    protected final Tensor gate(Tensor projection, int index) {
        return projection.slice(1, index * hiddenSize, (index + 1) * hiddenSize);
    }
}
