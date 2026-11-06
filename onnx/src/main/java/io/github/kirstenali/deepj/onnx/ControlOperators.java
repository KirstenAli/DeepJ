package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Map;

final class ControlOperators {
    private ControlOperators() {}

    static void register(OnnxRegistry r) {
        r.add("If", 1, "then_branch else_branch", ControlOperators::conditional);
        r.add("Loop", 1, "body", ControlOperators::loop);
    }

    private static List<OnnxValue> conditional(OnnxContext c) {
        if (c.input(0).elementType() != 9)
            throw new IllegalArgumentException("If requires BOOL condition");
        return c.session.run(c.graph(c.input(0).scalar() == 0 ? "else_branch" : "then_branch"), Map.of(), c.values);
    }

    private static List<OnnxValue> loop(OnnxContext c) {
        return new LoopExecution(c).run();
    }

    static OnnxValue stackScan(List<OnnxValue> values, OnnxContext c, int index) {
        if (!values.isEmpty()) {
            var expanded = values.stream().map(v -> ValueOps.reshape(v, ValueOps.join(new int[] {1}, v.shape())))
                            .toList();
            return ValueOps.concatenate(expanded, 0);
        }
        int state = c.inputCount() - 2;
        var type = c.graph("body").getOutput(1 + state + index).getType().getTensorType();
        int[] shape = ValueOps.join(new int[] {0}, type.getShape().getDimList().stream().mapToInt(d -> {
                                            if (!d.hasDimValue())
                                                throw new IllegalArgumentException(
                                                        "Zero-iteration scan needs declared output"
                                                                + " dimensions");
                                            return Math.toIntExact(d.getDimValue());
                                        }).toArray());
        if (OnnxTensors.floating(type.getElemType()))
            return CastOperators.cast(OnnxValue.of(
                            Tensor.zeros(io.github.kirstenali.deepj.tensor.DType.FLOAT32, shape)), type.getElemType());
        return OnnxValue.host(new long[0], shape, type.getElemType());
    }
}
