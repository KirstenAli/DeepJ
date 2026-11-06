package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import java.util.Map;

final class InputChecks {
    private InputChecks() {}

    static void validate(Onnx.ValueInfoProto info, OnnxValue value, Map<String, Integer> symbols) {
        if (!info.hasType()) return;
        var type = info.getType().getTensorType();
        if (type.getElemType() != 0 && value.elementType() != type.getElemType()) {
            throw new IllegalArgumentException("Value element type differs: " + info.getName());
        }
        if (!type.hasShape()) return;
        int[] actual = value.shape();
        if (type.getShape().getDimCount() != actual.length)
            throw new IllegalArgumentException("Value rank differs: " + info.getName());
        for (int axis = 0; axis < actual.length; axis++) {
            dimension(type.getShape().getDim(axis), actual[axis], symbols);
        }
    }

    private static void dimension(
            Onnx.TensorShapeProto.Dimension dimension, int actual, Map<String, Integer> symbols) {
        if (dimension.hasDimValue() && dimension.getDimValue() != actual) {
            throw new IllegalArgumentException("ONNX input dimension differs from declaration");
        }
        if (!dimension.getDimParam().isEmpty()) {
            Integer previous = symbols.putIfAbsent(dimension.getDimParam(), actual);
            if (previous != null && previous != actual)
                throw new IllegalArgumentException("ONNX symbolic dimensions differ");
        }
    }
}
