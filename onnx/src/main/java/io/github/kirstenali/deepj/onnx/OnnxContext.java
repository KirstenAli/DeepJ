package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Map;

/** One node invocation. Missing optional inputs retain their ONNX positions. */
public final class OnnxContext {
    final Onnx.NodeProto node;
    final Map<String, OnnxValue> values;
    final OnnxSession session;
    final Map<String, Onnx.AttributeProto> attributes;
    private final long version;

    OnnxContext(Onnx.NodeProto node, Map<String, OnnxValue> values, OnnxSession session, long version) {
        this.node = node;
        this.values = values;
        this.session = session;
        this.version = version;
        attributes = node.getAttributeList().stream().collect(java.util.stream.Collectors.toUnmodifiableMap(
                                        Onnx.AttributeProto::getName, a -> a));
    }

    public String operator() {
        return node.getOpType();
    }

    public long opset() {
        return version;
    }

    public int inputCount() {
        return node.getInputCount();
    }

    public int outputCount() {
        return node.getOutputCount();
    }

    public boolean hasInput(int i) {
        return i < inputCount() && !node.getInput(i).isEmpty();
    }

    public boolean hasAttribute(String name) {
        return attributes.containsKey(name);
    }

    public OnnxValue input(int i) {
        if (!hasInput(i)) throw new IllegalArgumentException("Missing ONNX input " + i);
        OnnxValue value = values.get(node.getInput(i));
        if (value == null)
            throw new IllegalArgumentException("Unknown ONNX value " + node.getInput(i));
        return value;
    }

    public Tensor tensor(int i) {
        return input(i).tensor();
    }

    public int integer(String name, int fallback) {
        return Math.toIntExact(longAttribute(name, fallback));
    }

    public long longAttribute(String name, long fallback) {
        return hasAttribute(name) ? attributes.get(name).getI() : fallback;
    }

    public float floating(String name, float fallback) {
        return hasAttribute(name) ? attributes.get(name).getF() : fallback;
    }

    public String string(String name, String fallback) {
        return hasAttribute(name) ? attributes.get(name).getS().toStringUtf8() : fallback;
    }

    public int[] integers(String name, int... fallback) {
        return hasAttribute(name) ? attributes.get(name).getIntsList().stream().mapToInt(Math::toIntExact).toArray()
                : fallback.clone();
    }

    public List<String> strings(String name) {
        return hasAttribute(name) ? attributes.get(name).getStringsList().stream()
                        .map(com.google.protobuf.ByteString::toStringUtf8).toList() : List.of();
    }

    public Onnx.TensorProto tensorAttribute(String name) {
        return attributes.get(name).getT();
    }

    Onnx.GraphProto graph(String name) {
        return attributes.get(name).getG();
    }

    public OnnxValue floatingResult(Tensor result) {
        return OnnxValue.of(result);
    }

    List<OnnxValue> one(Tensor result) {
        return List.of(OnnxValue.of(result));
    }

    int[] axes(int input, int... fallback) {
        return hasInput(input) ? input(input).ints() : integers("axes", fallback);
    }
}
