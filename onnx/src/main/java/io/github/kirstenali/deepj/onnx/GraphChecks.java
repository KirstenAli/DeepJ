package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class GraphChecks {
    private GraphChecks() {}

    static void validate(Onnx.GraphProto graph, Set<String> captures, List<String> problems) {
        var available = new HashSet<>(captures);
        var inputs = new HashSet<String>();
        graph.getInputList().forEach(i -> define(inputs, i.getName(), problems));
        available.addAll(inputs);
        var weights = new HashSet<String>();
        graph.getInitializerList().forEach(t -> define(weights, t.getName(), problems));
        available.addAll(weights);
        for (var node : graph.getNodeList()) {
            checkNode(node, available, problems);
        }
        for (var output : graph.getOutputList()) {
            if (!available.contains(output.getName()))
                problems.add("Undefined ONNX graph output " + output.getName());
        }
    }

    private static void checkNode(
            Onnx.NodeProto node, Set<String> available, List<String> problems) {
        for (String input : node.getInputList()) {
            if (!input.isEmpty() && !available.contains(input))
                problems.add("Undefined ONNX input " + input);
        }
        for (var attribute : node.getAttributeList()) {
            if (attribute.hasG()) validate(attribute.getG(), available, problems);
        }
        for (String output : node.getOutputList()) {
            if (!output.isEmpty()) define(available, output, problems);
        }
    }

    private static void define(Set<String> values, String name, List<String> problems) {
        if (name.isEmpty() || !values.add(name))
            problems.add("Empty or duplicate ONNX graph value: " + name);
    }
}
