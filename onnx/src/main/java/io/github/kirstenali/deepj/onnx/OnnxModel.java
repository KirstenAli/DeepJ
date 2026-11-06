package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** An imported computation graph, executed with the currently selected DeepJ backend. */
public final class OnnxModel implements Layer, AutoCloseable {
    private final Onnx.ModelProto model;
    private final OnnxSession session;
    private final OnnxInspection inspection;
    private final Map<String, Parameter> parameters = new LinkedHashMap<>();
    private final TensorBackend backend;
    private boolean closed;

    private OnnxModel(Onnx.ModelProto model, OnnxSession session, OnnxInspection inspection) {
        this.model = model;
        this.session = session;
        this.inspection = inspection;
        backend = Tensor.backend();
    }

    public static OnnxModel load(Path path) throws IOException {
        return load(path, OnnxRegistry.standard(), OnnxOptions.defaults());
    }

    public static OnnxModel load(Path path, OnnxRegistry registry, OnnxOptions options)
            throws IOException {
        var model = OnnxFunctions.expand(OnnxLoader.parse(path, options));
        var inspection = OnnxLoader.inspect(model, registry);
        if (!inspection.supported())
            throw new IOException(String.join("\n", inspection.problems()));
        var session = new OnnxSession(path.toAbsolutePath().getParent(), options, inspection.opsets(), registry);
        try {
            session.prepare(model.getGraph());
            return new OnnxModel(model, session, inspection);
        } catch (IOException | RuntimeException | Error failure) {
            session.close();
            throw failure;
        }
    }

    public static OnnxInspection inspect(Path path) throws IOException {
        return OnnxLoader.inspect(OnnxFunctions.expand(OnnxLoader.parse(path, OnnxOptions.defaults())),
                OnnxRegistry.standard());
    }

    public OnnxInspection inspection() {
        return inspection;
    }

    public Map<String, OnnxValue> run(Map<String, OnnxValue> inputs) {
        requireOpen();
        var declared = new java.util.HashSet<>(inspection.inputs());
        if (!declared.containsAll(inputs.keySet()))
            throw new IllegalArgumentException("Unknown ONNX graph input");
        var results = session.run(model.getGraph(), Map.copyOf(inputs), Map.of());
        var output = new LinkedHashMap<String, OnnxValue>();
        for (int i = 0; i < results.size(); i++) {
            output.put(inspection.outputs().get(i), results.get(i));
        }
        return java.util.Collections.unmodifiableMap(output);
    }

    public Tensor forward(Tensor input) {
        var names = inspection.inputs().stream().filter(n -> !initializerNames().contains(n)).toList();
        if (names.size() != 1 || inspection.outputs().size() != 1) {
            throw new IllegalStateException("Use named ONNX inputs and outputs for this graph");
        }
        return run(Map.of(names.get(0), OnnxValue.of(input))).get(inspection.outputs().get(0)).tensor();
    }

    public List<Parameter> trainable(String... names) throws IOException {
        requireOpen();
        var initializers = session.initializers(model.getGraph());
        for (String name : names) {
            OnnxValue value = initializers.get(name);
            if (value == null || !value.isFloating())
                throw new IllegalArgumentException("Not a floating initializer: " + name);
            parameters.computeIfAbsent(name, key -> new Parameter(key, value.tensor()));
        }
        return parameters();
    }

    public List<Parameter> parameters() {
        return List.copyOf(parameters.values());
    }

    public void zeroGrad() {
        parameters.values().forEach(Parameter::zeroGrad);
    }

    private java.util.Set<String> initializerNames() {
        return model.getGraph().getInitializerList().stream().map(Onnx.TensorProto::getName)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    private void requireOpen() {
        if (closed) throw new IllegalStateException("ONNX model is closed");
        if (backend != Tensor.backend())
            throw new IllegalStateException("Use the backend selected when loading ONNX");
    }

    @Override
    public void close() {
        if (!closed) session.close();
        closed = true;
    }
}
