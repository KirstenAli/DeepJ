package io.github.kirstenali.deepj.onnx;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/** State belongs to one Loop invocation, including its per-output scan accumulators. */
final class LoopExecution {
    private final OnnxContext context;
    private final List<OnnxValue> state = new ArrayList<>();
    private final List<List<OnnxValue>> scans = new ArrayList<>();

    LoopExecution(OnnxContext context) {
        this.context = context;
        for (int i = 2; i < context.inputCount(); i++) {
            state.add(context.input(i));
        }
        int count = context.outputCount() - state.size();
        if (count < 0
                || context.graph("body").getInputCount() != 2 + state.size()
                || context.graph("body").getOutputCount() != 1 + state.size() + count) {
            throw new IllegalArgumentException("Loop body input/output counts differ");
        }
        for (int i = 0; i < count; i++) {
            scans.add(new ArrayList<>());
        }
    }

    List<OnnxValue> run() {
        long maximum = context.hasInput(0) ? context.input(0).scalar() : Long.MAX_VALUE;
        boolean condition = !context.hasInput(1) || context.input(1).scalar() != 0;
        for (long iteration = 0; iteration < maximum && condition; iteration++) {
            condition = step(iteration, condition);
        }
        var results = new ArrayList<>(state);
        for (int i = 0; i < scans.size(); i++) {
            results.add(ControlOperators.stackScan(scans.get(i), context, i));
        }
        return results;
    }

    private boolean step(long iteration, boolean condition) {
        if (iteration >= context.session.options.maxLoopIterations()) {
            throw new IllegalArgumentException("ONNX Loop iteration limit exceeded");
        }
        var graph = context.graph("body");
        var arguments = new LinkedHashMap<String, OnnxValue>();
        arguments.put(graph.getInput(0).getName(), ValueOps.scalar(iteration));
        arguments.put(graph.getInput(1).getName(), ValueOps.bool(condition));
        for (int i = 0; i < state.size(); i++) {
            arguments.put(graph.getInput(i + 2).getName(), state.get(i));
        }
        var result = context.session.run(graph, arguments, context.values);
        update(result);
        if (result.get(0).elementType() != 9)
            throw new IllegalArgumentException("Loop condition must be BOOL");
        return result.get(0).scalar() != 0;
    }

    private void update(List<OnnxValue> result) {
        for (int i = 0; i < state.size(); i++) {
            ValueOps.sameType(state.get(i), result.get(i + 1));
            state.set(i, result.get(i + 1));
        }
        for (int i = 0; i < scans.size(); i++) {
            scans.get(i).add(result.get(1 + state.size() + i));
        }
    }
}
