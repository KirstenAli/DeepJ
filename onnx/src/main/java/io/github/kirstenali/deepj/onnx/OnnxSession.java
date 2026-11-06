package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

final class OnnxSession implements AutoCloseable {
    final OnnxOptions options;
    private final Path directory;
    private final Map<String, Long> versions;
    private final OnnxRegistry registry;
    private final Map<Onnx.TensorProto, OnnxValue> tensors = new IdentityHashMap<>();
    private final Map<Onnx.GraphProto, Map<String, OnnxValue>> constants = new IdentityHashMap<>();
    private long weightBytes;
    private final Map<Onnx.NodeProto, Random> randoms = new IdentityHashMap<>();

    OnnxSession(
            Path directory,
            OnnxOptions options,
            Map<String, Long> versions,
            OnnxRegistry registry) {
        this.directory = directory;
        this.options = options;
        this.versions = versions;
        this.registry = registry;
    }

    void prepare(Onnx.GraphProto graph) throws IOException {
        initializers(graph);
        for (var node : graph.getNodeList()) {
            for (var attribute : node.getAttributeList()) {
                if (attribute.hasT()) read(attribute.getT());
                if (attribute.hasG()) prepare(attribute.getG());
            }
        }
    }

    Map<String, OnnxValue> initializers(Onnx.GraphProto graph) throws IOException {
        if (constants.containsKey(graph)) return constants.get(graph);
        var loaded = new LinkedHashMap<String, OnnxValue>();
        for (var tensor : graph.getInitializerList()) {
            if (loaded.putIfAbsent(tensor.getName(), read(tensor)) != null) {
                throw new IOException("Duplicate ONNX initializer: " + tensor.getName());
            }
        }
        constants.put(graph, Map.copyOf(loaded));
        return constants.get(graph);
    }

    OnnxValue read(Onnx.TensorProto tensor) throws IOException {
        if (tensors.containsKey(tensor)) return tensors.get(tensor);
        int[] shape = tensor.getDimsList().stream().mapToInt(Math::toIntExact).toArray();
        long elements = OnnxValue.elements(shape);
        long width =
                OnnxTensors.floating(tensor.getDataType()) ? tensor.getDataType() == 1 ? 4 : 2 : 8;
        weightBytes = Math.addExact(weightBytes, Math.multiplyExact(elements, width));
        if (weightBytes > options.maxWeightBytes())
            throw new IOException("ONNX model exceeds weight budget");
        OnnxValue value = OnnxTensors.read(tensor, directory, options);
        tensors.put(tensor, value);
        return value;
    }

    Random random(Onnx.NodeProto node, long seed) {
        return randoms.computeIfAbsent(node, ignored -> new Random(seed));
    }

    OnnxValue constant(Onnx.TensorProto tensor) {
        try {
            return read(tensor);
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    List<OnnxValue> run(
            Onnx.GraphProto graph,
            Map<String, OnnxValue> arguments,
            Map<String, OnnxValue> captures) {
        try (var scope = TensorScope.open()) {
            var environment = environment(graph, arguments, captures);
            var values = environment.values();
            for (var node : graph.getNodeList()) {
                execute(node, values);
            }
            var output = new ArrayList<OnnxValue>();
            for (var value : graph.getOutputList()) {
                OnnxValue result = require(values, value.getName());
                InputChecks.validate(value, result, environment.dimensions());
                if (result.isFloating()) scope.escape(result.tensor());
                output.add(result);
            }
            return List.copyOf(output);
        }
    }

    private Environment environment(
            Onnx.GraphProto graph,
            Map<String, OnnxValue> arguments,
            Map<String, OnnxValue> captures) {
        var values = new HashMap<>(captures);
        values.putAll(constants.getOrDefault(graph, Map.of()));
        values.putAll(arguments);
        var dimensions = new HashMap<String, Integer>();
        for (var input : graph.getInputList()) {
            InputChecks.validate(input, require(values, input.getName()), dimensions);
        }
        return new Environment(values, dimensions);
    }

    private void execute(Onnx.NodeProto node, Map<String, OnnxValue> values) {
        try {
            String domain = OnnxRegistry.domain(node.getDomain());
            var context = new OnnxContext(node, values, this, versions.get(domain));
            OnnxSchemas.validate(context);
            List<OnnxValue> output =
                    registry.find(domain, node.getOpType()).operator().apply(context);
            bind(node, output, values);
        } catch (RuntimeException failure) {
            throw new IllegalArgumentException(
                    "ONNX "
                            + node.getOpType()
                            + " ["
                            + node.getName()
                            + "]: "
                            + failure.getMessage(),
                    failure);
        }
    }

    private static void bind(
            Onnx.NodeProto node, List<OnnxValue> output, Map<String, OnnxValue> values) {
        for (int i = 0; i < node.getOutputCount(); i++) {
            String name = node.getOutput(i);
            if (name.isEmpty()) continue;
            if (i >= output.size())
                throw new IllegalArgumentException("Unsupported ONNX output position " + i);
            if (values.putIfAbsent(name, Objects.requireNonNull(output.get(i))) != null) {
                throw new IllegalArgumentException("Duplicate ONNX output: " + name);
            }
        }
    }

    private static OnnxValue require(Map<String, OnnxValue> values, String name) {
        OnnxValue value = values.get(name);
        if (value == null) throw new IllegalArgumentException("Missing ONNX graph value: " + name);
        return value;
    }

    @Override
    public void close() {
        tensors.values().stream()
                .filter(OnnxValue::isFloating)
                .forEach(v -> v.tensor().releaseDeviceBufferRetention());
        tensors.clear();
        constants.clear();
        randoms.clear();
    }

    private record Environment(Map<String, OnnxValue> values, Map<String, Integer> dimensions) {}
}
