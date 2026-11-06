package io.github.kirstenali.deepj.onnx;

import java.util.List;
import java.util.Map;

/** Import compatibility is explicit for each graph and its declared operator sets. */
public record OnnxInspection(
        long irVersion,
        Map<String, Long> opsets,
        List<String> inputs,
        List<String> outputs,
        List<String> problems) {
    public OnnxInspection {
        opsets = Map.copyOf(opsets);
        inputs = List.copyOf(inputs);
        outputs = List.copyOf(outputs);
        problems = List.copyOf(problems);
    }

    public boolean supported() {
        return problems.isEmpty();
    }
}
