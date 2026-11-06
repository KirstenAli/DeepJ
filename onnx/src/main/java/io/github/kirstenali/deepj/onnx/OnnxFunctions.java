package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Expands model-local functions without changing graph or parameter names. */
final class OnnxFunctions {
    private final Map<String, Onnx.FunctionProto> functions = new HashMap<>();
    private final Map<String, Long> versions = new LinkedHashMap<>();
    private int serial;
    private final java.util.Set<String> occupied = new java.util.HashSet<>();

    private OnnxFunctions(Onnx.ModelProto model) {
        versions.putAll(OnnxLoader.opsets(model));
        var graphNames = new HashMap<String, String>();
        collectGraphNames(model.getGraph(), graphNames, "");
        occupied.addAll(graphNames.keySet());
        for (var function : model.getFunctionsList()) {
            String key = key(function.getDomain(), function.getName(), function.getOverload());
            if (functions.putIfAbsent(key, function) != null) {
                throw new IllegalArgumentException("Duplicate ONNX local function: " + key);
            }
        }
    }

    static Onnx.ModelProto expand(Onnx.ModelProto model) {
        var expander = new OnnxFunctions(model);
        var result = model.toBuilder().setGraph(expander.graph(model.getGraph(), Map.of(), Map.of(), 0));
        result.clearOpsetImport().clearFunctions();
        expander.versions.forEach((domain, version) -> result.addOpsetImport(Onnx.OperatorSetIdProto.newBuilder()
                                        .setDomain(domain).setVersion(version)));
        return result.build();
    }

    private Onnx.GraphProto graph(Onnx.GraphProto graph, Map<String, String> names,
            Map<String, Onnx.AttributeProto> attributes, int depth) {
        var result = renameGraph(graph, names).clearNode();
        for (var node : graph.getNodeList()) {
            for (var expanded : node(rename(node, names, attributes, depth), depth)) {
                result.addNode(expanded);
            }
        }
        return result.build();
    }

    private List<Onnx.NodeProto> node(Onnx.NodeProto call, int depth) {
        if (depth > 32)
            throw new IllegalArgumentException("ONNX local function nesting exceeds 32");
        var function = functions.get(key(call.getDomain(), call.getOpType(), call.getOverload()));
        if (function == null) return List.of(call);
        mergeVersions(function);
        var names = names(function, call);
        var attributes = attributes(function, call);
        var result = new ArrayList<Onnx.NodeProto>();
        for (var operation : function.getNodeList()) {
            result.addAll(node(rename(operation, names, attributes, depth + 1), depth + 1));
        }
        appendAliases(result, function, call, names);
        return result;
    }

    private void mergeVersions(Onnx.FunctionProto function) {
        for (var version : function.getOpsetImportList()) {
            Long previous = versions.putIfAbsent(OnnxRegistry.domain(version.getDomain()), version.getVersion());
            if (previous != null && previous != version.getVersion()) {
                throw new IllegalArgumentException(
                        "Conflicting opset versions in local function " + function.getName());
            }
        }
    }

    private Map<String, String> names(Onnx.FunctionProto function, Onnx.NodeProto call) {
        if (call.getInputCount() != function.getInputCount()
                || call.getOutputCount() != function.getOutputCount()) {
            throw new IllegalArgumentException("Local function argument count differs: " + function.getName());
        }
        String prefix = freshPrefix();
        var result = new HashMap<String, String>();
        for (int i = 0; i < call.getInputCount(); i++) {
            result.put(function.getInput(i), call.getInput(i));
        }
        for (int i = 0; i < call.getOutputCount(); i++) {
            String name = function.getOutput(i);
            result.putIfAbsent(name, call.getOutput(i).isEmpty() ? prefix + name : call.getOutput(i));
        }
        function.getNodeList().forEach(n -> collectNames(n, result, prefix));
        return result;
    }

    private static Map<String, Onnx.AttributeProto> attributes(Onnx.FunctionProto function, Onnx.NodeProto call) {
        var result = new HashMap<String, Onnx.AttributeProto>();
        function.getAttributeProtoList().forEach(a -> result.put(a.getName(), a));
        call.getAttributeList().forEach(a -> result.put(a.getName(), a));
        for (String required : function.getAttributeList()) {
            if (!result.containsKey(required))
                throw new IllegalArgumentException("Missing function attribute " + required);
        }
        return result;
    }

    private Onnx.NodeProto rename(Onnx.NodeProto node, Map<String, String> names,
            Map<String, Onnx.AttributeProto> attributes, int depth) {
        var result = node.toBuilder().clearInput().clearOutput().clearAttribute();
        node.getInputList().forEach(v -> result.addInput(names.getOrDefault(v, v)));
        node.getOutputList().forEach(v -> result.addOutput(names.getOrDefault(v, v)));
        for (var attribute : node.getAttributeList()) {
            var value = substitute(attribute, attributes);
            if (value.hasG())
                value = value.toBuilder().setG(graph(value.getG(), names, attributes, depth)).build();
            if (value.getGraphsCount() != 0)
                throw new IllegalArgumentException("Graph-list attributes are unsupported");
            result.addAttribute(value);
        }
        return result.build();
    }

    private static Onnx.AttributeProto substitute(
            Onnx.AttributeProto attribute, Map<String, Onnx.AttributeProto> values) {
        if (attribute.getRefAttrName().isEmpty()) return attribute;
        var value = values.get(attribute.getRefAttrName());
        if (value == null)
            throw new IllegalArgumentException("Unresolved function attribute " + attribute.getRefAttrName());
        return value.toBuilder().setName(attribute.getName()).clearRefAttrName().build();
    }

    private static String key(String domain, String name, String overload) {
        return OnnxRegistry.domain(domain) + ":" + name + ":" + overload;
    }

    private static Onnx.GraphProto.Builder renameGraph(Onnx.GraphProto graph, Map<String, String> names) {
        var result = graph.toBuilder().clearInput().clearOutput().clearValueInfo().clearInitializer();
        graph.getInputList().forEach(v -> result.addInput(renameValue(v, names)));
        graph.getOutputList().forEach(v -> result.addOutput(renameValue(v, names)));
        graph.getValueInfoList().forEach(v -> result.addValueInfo(renameValue(v, names)));
        graph.getInitializerList().forEach(t -> result.addInitializer(t.toBuilder().setName(names.getOrDefault(
                                                                t.getName(), t.getName()))));
        return result;
    }

    private static Onnx.ValueInfoProto renameValue(Onnx.ValueInfoProto value, Map<String, String> names) {
        return value.toBuilder().setName(names.getOrDefault(value.getName(), value.getName())).build();
    }

    private static void collectNames(Onnx.NodeProto node, Map<String, String> names, String prefix) {
        node.getOutputList().stream().filter(v -> !v.isEmpty()).forEach(v -> names.putIfAbsent(v, prefix + v));
        for (var attribute : node.getAttributeList()) {
            if (attribute.hasG()) collectGraphNames(attribute.getG(), names, prefix);
        }
    }

    private static void collectGraphNames(Onnx.GraphProto graph, Map<String, String> names, String prefix) {
        graph.getInputList().forEach(v -> names.putIfAbsent(v.getName(), prefix + v.getName()));
        graph.getInitializerList().forEach(t -> names.putIfAbsent(t.getName(), prefix + t.getName()));
        graph.getNodeList().forEach(n -> collectNames(n, names, prefix));
    }

    private String freshPrefix() {
        String prefix;
        do {
            prefix = "__deepj_function_" + serial++ + "_";
        } while (hasPrefix(prefix));
        return prefix;
    }

    private boolean hasPrefix(String prefix) {
        return occupied.stream().anyMatch(n -> n.startsWith(prefix));
    }

    private static void appendAliases(List<Onnx.NodeProto> result, Onnx.FunctionProto function, Onnx.NodeProto call,
            Map<String, String> names) {
        for (int i = 0; i < call.getOutputCount(); i++) {
            String source = names.get(function.getOutput(i)), target = call.getOutput(i);
            if (!target.isEmpty() && !target.equals(source)) {
                result.add(Onnx.NodeProto.newBuilder().setOpType("Identity").addInput(source).addOutput(target)
                                .build());
            }
        }
    }
}
