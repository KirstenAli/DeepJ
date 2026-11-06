package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.nn.TensorJoin;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

final class RecurrentOperators {
    private RecurrentOperators() {}

    static void register(OnnxRegistry r) {
        String common = "activation_alpha activation_beta activations clip direction hidden_size layout";
        r.add("RNN", 7, common, RecurrentOperators::run);
        r.add("GRU", 7, common + " linear_before_reset", RecurrentOperators::run);
        r.add("LSTM", 7, common + " input_forget", RecurrentOperators::run);
    }

    private static List<OnnxValue> run(OnnxContext c) {
        var r = new RecurrentInputs(c);
        var results = new ArrayList<Result>();
        for (int direction = 0; direction < r.directions; direction++) {
            results.add(direction(r, direction));
        }
        Tensor y = TensorJoin.stack(1, results.stream().map(Result::sequence).toArray(Tensor[]::new));
        Tensor h = TensorJoin.stack(0, results.stream().map(v -> v.state().hidden()).toArray(Tensor[]::new));
        Tensor cell = TensorJoin.stack(0, results.stream().map(v -> v.state().cell()).toArray(Tensor[]::new));
        return r.layout == 0 ? values(y, h, cell)
                : values(y.permute(2, 0, 1, 3), h.transpose(0, 1), cell.transpose(0, 1));
    }

    private static Result direction(RecurrentInputs r, int direction) {
        var state = new RecurrentCell.State(r.initial.select(0, direction), r.cell.select(0, direction));
        Tensor zero = Tensor.backend().zeros(r.input.dtype(), r.input.size(1), r.hidden);
        Tensor[] outputs = new Tensor[r.input.size(0)];
        for (int step = 0; step < outputs.length; step++) {
            int time = r.reverse(direction) ? outputs.length - step - 1 : step;
            var next = RecurrentCell.step(r, r.input.select(0, time), state, direction);
            state = new RecurrentCell.State(r.masked(next.hidden(), state.hidden(), time),
                            r.masked(next.cell(), state.cell(), time));
            outputs[time] = r.masked(state.hidden(), zero, time);
        }
        Tensor sequence = outputs.length == 0 ? Tensor.backend().zeros(r.input.dtype(), 0, r.input.size(1), r.hidden)
                        : TensorJoin.stack(0, outputs);
        return new Result(sequence, state);
    }

    private record Result(Tensor sequence, RecurrentCell.State state) {}

    private static List<OnnxValue> values(Tensor sequence, Tensor hidden, Tensor cell) {
        return List.of(OnnxValue.of(sequence), OnnxValue.of(hidden), OnnxValue.of(cell));
    }
}
