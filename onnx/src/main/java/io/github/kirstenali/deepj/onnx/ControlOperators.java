package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.Map;

final class ControlOperators {
    private ControlOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add("If", 1, "then_branch else_branch", ControlOperators::conditional);
        registry.add("Loop", 1, "body", ControlOperators::loop);
    }

    private static List<OnnxValue> conditional(OnnxContext context) {
        if (context.input(0).elementType() != 9)
            throw new IllegalArgumentException("If requires BOOL condition");
        return context.session.run(
                context.graph(context.input(0).scalar() == 0 ? "else_branch" : "then_branch"),
                Map.of(),
                context.values);
    }

    private static List<OnnxValue> loop(OnnxContext context) {
        return new LoopExecution(context).run();
    }

    static OnnxValue stackScan(List<OnnxValue> values, OnnxContext context, int index) {
        if (values.isEmpty()) return emptyScan(context, index);
        var expanded =
                values.stream()
                        .map(
                                value ->
                                        ValueOps.reshape(
                                                value, ValueOps.join(new int[] {1}, value.shape())))
                        .toList();
        return ValueOps.concatenate(expanded, 0);
    }

    private static OnnxValue emptyScan(OnnxContext context, int index) {
        int state = context.inputCount() - 2;
        var type = context.graph("body").getOutput(1 + state + index).getType().getTensorType();
        int[] shape = ValueOps.join(new int[] {0}, declaredDimensions(type.getShape()));
        if (OnnxTensors.floating(type.getElemType())) {
            return CastOperators.cast(
                    OnnxValue.of(Tensor.zeros(DType.FLOAT32, shape)), type.getElemType());
        }
        return OnnxValue.host(new long[0], shape, type.getElemType());
    }

    private static int[] declaredDimensions(Onnx.TensorShapeProto shape) {
        return shape.getDimList().stream().mapToInt(ControlOperators::declaredDimension).toArray();
    }

    private static int declaredDimension(Onnx.TensorShapeProto.Dimension dimension) {
        if (!dimension.hasDimValue())
            throw new IllegalArgumentException(
                    "Zero-iteration scan needs declared output dimensions");
        return Math.toIntExact(dimension.getDimValue());
    }
}
