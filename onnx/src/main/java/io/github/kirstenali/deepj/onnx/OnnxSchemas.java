package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

/** Official versioned contracts supplement the implementation's supported-attribute checks. */
final class OnnxSchemas {
    private static final Map<String, NavigableMap<Long, Contract>> CONTRACTS = read();

    private OnnxSchemas() {}

    static void inspect(Onnx.NodeProto node, long version, List<String> problems) {
        Contract contract = find(node.getOpType(), version);
        if (contract == null) {
            problems.add("No ONNX schema for " + node.getOpType() + " at opset " + version);
            return;
        }
        if (!contract.inputs.contains(node.getInputCount())
                || !contract.outputs.contains(node.getOutputCount())) {
            problems.add("Invalid input/output count for ONNX " + node.getOpType());
        }
        var supplied = supplied(node, problems);
        contract.attributes.forEach(
                (name, attribute) -> attribute.check(name, supplied.get(name), problems));
        supplied.forEach(
                (name, value) -> {
                    if (!contract.attributes.containsKey(name))
                        problems.add("Invalid opset attribute " + name);
                });
        requiredInputs(node, contract, problems);
    }

    private static void requiredInputs(
            Onnx.NodeProto node, Contract contract, List<String> problems) {
        for (int i = 0; i < contract.formals.size(); i++) {
            if (contract.formals.get(i).option == 0
                    && (i >= node.getInputCount() || node.getInput(i).isEmpty())) {
                problems.add("Missing required ONNX input " + i + " for " + node.getOpType());
            }
        }
    }

    static void validate(OnnxContext context) {
        if (!OnnxRegistry.domain(context.node.getDomain()).isEmpty()) return;
        Contract contract = find(context.operator(), context.opset());
        var bound = new HashMap<String, Integer>();
        for (int i = 0; i < context.inputCount(); i++) {
            if (!context.hasInput(i)) continue;
            Formal formal = contract.formals.get(Math.min(i, contract.formals.size() - 1));
            formal.check(context.input(i).elementType(), bound);
        }
    }

    private static Contract find(String operator, long version) {
        var schemas = CONTRACTS.get(operator);
        var entry = schemas == null ? null : schemas.floorEntry(version);
        return entry == null ? null : entry.getValue();
    }

    private static Map<String, NavigableMap<Long, Contract>> read() {
        var result = new HashMap<String, NavigableMap<Long, Contract>>();
        try (var stream = OnnxSchemas.class.getResourceAsStream("schemas.tsv");
                var reader =
                        new BufferedReader(
                                new InputStreamReader(
                                        Objects.requireNonNull(stream), StandardCharsets.UTF_8))) {
            for (String line : reader.lines().filter(l -> !l.startsWith("#")).toList()) {
                String[] values = line.split("\\t", -1);
                result.computeIfAbsent(values[0], ignored -> new TreeMap<>())
                        .put(Long.parseLong(values[1]), contract(values));
            }
            return Map.copyOf(result);
        } catch (java.io.IOException failure) {
            throw new ExceptionInInitializerError(failure);
        }
    }

    private static Contract contract(String[] values) {
        return new Contract(
                new Count(values[2], values[3]),
                new Count(values[4], values[5]),
                schemaAttributes(values[6]),
                schemaFormals(values[7]));
    }

    private static List<String> entries(String text, String delimiter) {
        return text.isEmpty() || text.equals("-")
                ? List.of()
                : Arrays.asList(text.split(delimiter));
    }

    private record Count(int minimum, int maximum) {
        Count(String minimum, String maximum) {
            this(Integer.parseInt(minimum), Integer.parseInt(maximum));
        }

        boolean contains(int count) {
            return count >= minimum && count <= maximum;
        }
    }

    private record Contract(
            Count inputs, Count outputs, Map<String, Attribute> attributes, List<Formal> formals) {}

    private record Attribute(int type, boolean required) {
        void check(String name, Onnx.AttributeProto value, List<String> problems) {
            if (value == null && required) problems.add("Missing required ONNX attribute " + name);
            if (value != null && value.getTypeValue() != type)
                problems.add("Wrong ONNX attribute type at " + name);
        }
    }

    private record Formal(String variable, int option, Set<Integer> types, boolean homogeneous) {
        void check(int type, Map<String, Integer> bound) {
            if (!types.contains(type))
                throw new IllegalArgumentException(
                        "Invalid ONNX input type " + type + " for " + variable);
            if (!homogeneous) return;
            Integer previous = bound.putIfAbsent(variable, type);
            if (previous != null && previous != type)
                throw new IllegalArgumentException("ONNX type variable " + variable + " differs");
        }
    }

    private static Map<String, Onnx.AttributeProto> supplied(
            Onnx.NodeProto node, List<String> problems) {
        var result = new HashMap<String, Onnx.AttributeProto>();
        for (var attribute : node.getAttributeList()) {
            if (result.putIfAbsent(attribute.getName(), attribute) != null) {
                problems.add("Duplicate attribute " + attribute.getName());
            }
        }
        return result;
    }

    private static Map<String, Attribute> schemaAttributes(String text) {
        var attributes = new HashMap<String, Attribute>();
        for (String entry : entries(text, ",")) {
            String[] fields = entry.split(":");
            attributes.put(
                    fields[0], new Attribute(Integer.parseInt(fields[1]), fields[2].equals("1")));
        }
        return attributes;
    }

    private static List<Formal> schemaFormals(String text) {
        var formals = new ArrayList<Formal>();
        for (String entry : entries(text, ";")) {
            String[] fields = entry.split(":", -1);
            Set<Integer> types =
                    entries(fields[2], ",").stream()
                            .map(Integer::valueOf)
                            .collect(Collectors.toSet());
            formals.add(
                    new Formal(
                            fields[0], Integer.parseInt(fields[1]), types, fields[3].equals("1")));
        }
        return formals;
    }
}
