package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Rejects unsupported attribute values before allocating model weights. */
final class OperatorChecks {
    private OperatorChecks() {}

    static void inspect(Onnx.NodeProto node, List<String> problems) {
        var attributes = node.getAttributeList().stream().collect(Collectors.toMap(
                                        Onnx.AttributeProto::getName, a -> a, (a, b) -> a));
        switch (node.getOpType()) {
            case "Resize", "Upsample" -> resize(attributes, problems);
            case "Gelu" -> string(attributes, "approximate", Set.of("none", "tanh"), problems);
            case "Pad" -> string(attributes, "mode", Set.of("constant", "edge", "reflect", "wrap"), problems);
            case "Constant" -> constant(attributes, problems);
            default -> {}
        }
        integer(attributes, "stash_type", Set.of(1L), problems);
        integer(attributes, "storage_order", Set.of(0L), problems);
        booleanAttributes(attributes, problems);
    }

    private static void booleanAttributes(Map<String, Onnx.AttributeProto> attributes, List<String> problems) {
        for (String name : List.of("layout", "linear_before_reset", "input_forget", "keepdims", "ceil_mode",
                        "allowzero", "noop_with_empty_axes", "select_last_index", "largest", "sorted",
                        "training_mode")) {
            integer(attributes, name, Set.of(0L, 1L), problems);
        }
    }

    private static void resize(Map<String, Onnx.AttributeProto> attributes, List<String> problems) {
        integer(attributes, "antialias", Set.of(0L), problems);
        string(attributes, "mode", Set.of("nearest", "linear"), problems);
        string(attributes, "keep_aspect_ratio_policy", Set.of("stretch"), problems);
        string(attributes, "coordinate_transformation_mode", Set.of("asymmetric", "half_pixel", "pytorch_half_pixel",
                        "align_corners", "tf_half_pixel_for_nn", "half_pixel_symmetric"), problems);
        string(attributes, "nearest_mode", Set.of("floor", "ceil", "round_prefer_floor", "round_prefer_ceil"),
                problems);
    }

    private static void constant(Map<String, Onnx.AttributeProto> attributes, List<String> problems) {
        long count = attributes.keySet().stream().filter(k -> k.equals("value") || k.startsWith("value_")).count();
        if (count != 1) problems.add("Constant must declare exactly one value attribute");
    }

    private static void integer(Map<String, Onnx.AttributeProto> attributes, String name, Set<Long> accepted,
            List<String> problems) {
        var value = attributes.get(name);
        if (value != null && !accepted.contains(value.getI()))
            problems.add("Unsupported " + name + "=" + value.getI());
    }

    private static void string(Map<String, Onnx.AttributeProto> attributes, String name, Set<String> accepted,
            List<String> problems) {
        var value = attributes.get(name);
        if (value != null && !accepted.contains(value.getS().toStringUtf8())) {
            problems.add("Unsupported " + name + "=" + value.getS().toStringUtf8());
        }
    }
}
