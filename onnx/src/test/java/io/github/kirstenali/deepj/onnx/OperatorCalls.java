package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class OperatorCalls {
    private OperatorCalls() {}

    static List<OnnxValue> run(
            String operator, OnnxValue[] inputs, Onnx.AttributeProto... attributes) {
        var registry = OnnxRegistry.standard();
        var values = new LinkedHashMap<String, OnnxValue>();
        var node =
                Onnx.NodeProto.newBuilder()
                        .setOpType(operator)
                        .addAllAttribute(List.of(attributes));
        for (int index = 0; index < inputs.length; index++) {
            String name = "input" + index;
            node.addInput(name);
            values.put(name, inputs[index]);
        }
        try (var session =
                new OnnxSession(Path.of("."), OnnxOptions.defaults(), Map.of("", 18L), registry)) {
            var context = new OnnxContext(node.build(), values, session, 18);
            return registry.find("", operator).operator().apply(context);
        }
    }
}
