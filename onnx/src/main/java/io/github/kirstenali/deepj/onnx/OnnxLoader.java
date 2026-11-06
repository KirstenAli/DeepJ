package io.github.kirstenali.deepj.onnx;

import com.google.protobuf.CodedInputStream;
import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class OnnxLoader {
    private OnnxLoader() {}

    static Onnx.ModelProto parse(Path path, OnnxOptions options) throws IOException {
        if (Files.size(path) > options.maxModelBytes())
            throw new IOException("ONNX protobuf exceeds import limit");
        try (var stream = Files.newInputStream(path)) {
            var input = CodedInputStream.newInstance(stream);
            input.setSizeLimit(options.maxModelBytes());
            return Onnx.ModelProto.parseFrom(input);
        }
    }

    static Map<String, Long> opsets(Onnx.ModelProto model) {
        var versions = new LinkedHashMap<String, Long>();
        for (var entry : model.getOpsetImportList()) {
            if (versions.putIfAbsent(OnnxRegistry.domain(entry.getDomain()), entry.getVersion())
                    != null) {
                throw new IllegalArgumentException("Duplicate ONNX operator-set domain");
            }
        }
        return Map.copyOf(versions);
    }

    static OnnxInspection inspect(Onnx.ModelProto model, OnnxRegistry registry) {
        var problems = new ArrayList<String>();
        if (model.getIrVersion() < 3 || model.getIrVersion() > 14)
            problems.add("Supported ONNX IR versions: 3–14");
        if (!model.hasGraph()) problems.add("ONNX model has no graph");
        var versions = opsets(model);
        GraphChecks.validate(model.getGraph(), Set.of(), problems);
        inspectGraph(model.getGraph(), registry, versions, problems);
        return new OnnxInspection(
                model.getIrVersion(),
                versions,
                names(model.getGraph().getInputList()),
                names(model.getGraph().getOutputList()),
                problems);
    }

    private static List<String> names(List<Onnx.ValueInfoProto> values) {
        return values.stream().map(Onnx.ValueInfoProto::getName).toList();
    }

    private static void inspectGraph(
            Onnx.GraphProto graph,
            OnnxRegistry registry,
            Map<String, Long> versions,
            List<String> problems) {
        if (graph.getSparseInitializerCount() != 0)
            problems.add("Sparse initializers are unsupported: " + graph.getName());
        graph.getInitializerList().forEach(t -> checkType(t.getName(), t.getDataType(), problems));
        graph.getInputList().forEach(v -> checkValueType(v, problems));
        graph.getOutputList().forEach(v -> checkValueType(v, problems));
        for (var node : graph.getNodeList()) {
            inspectNode(node, registry, versions, problems);
        }
    }

    private static void checkValueType(Onnx.ValueInfoProto value, List<String> problems) {
        if (value.hasType() && value.getType().hasTensorType()) {
            checkType(value.getName(), value.getType().getTensorType().getElemType(), problems);
        } else if (value.hasType())
            problems.add("Unsupported non-tensor ONNX value: " + value.getName());
    }

    private static void checkType(String name, int type, List<String> problems) {
        if (!OnnxTensors.supported(type))
            problems.add("Unsupported ONNX element type " + type + " at " + name);
    }

    private static void inspectNode(
            Onnx.NodeProto node,
            OnnxRegistry registry,
            Map<String, Long> versions,
            List<String> problems) {
        String domain = OnnxRegistry.domain(node.getDomain());
        long version = versions.getOrDefault(domain, 0L);
        var rule = registry.find(domain, node.getOpType());
        if (domain.isEmpty()) {
            OnnxSchemas.inspect(node, version, problems);
            OperatorChecks.inspect(node, problems);
        }
        String label = domain + ":" + node.getOpType() + " [" + node.getName() + "]";
        if (rule == null) problems.add("Unsupported operator " + label);
        else if (version < rule.first() || version > rule.last())
            problems.add("Unsupported opset " + version + " for " + label);
        for (var attribute : node.getAttributeList()) {
            if (rule != null && !rule.attributes().contains(attribute.getName())) {
                problems.add("Unsupported attribute " + attribute.getName() + " for " + label);
            }
            inspectAttribute(attribute, registry, versions, problems);
        }
    }

    private static void inspectAttribute(
            Onnx.AttributeProto attribute,
            OnnxRegistry registry,
            Map<String, Long> versions,
            List<String> problems) {
        if (!attribute.getRefAttrName().isEmpty())
            problems.add("Unresolved ONNX function attribute");
        if (attribute.hasT())
            checkType(attribute.getName(), attribute.getT().getDataType(), problems);
        if (attribute.hasG()) inspectGraph(attribute.getG(), registry, versions, problems);
        attribute.getGraphsList().forEach(g -> inspectGraph(g, registry, versions, problems));
    }
}
