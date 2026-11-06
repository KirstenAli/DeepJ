package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

final class ReductionOperators {
    private ReductionOperators() {}

    static void register(OnnxRegistry r) {
        for (String op : List.of("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1",
                        "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare")) {
            r.add(op, 1, "axes keepdims noop_with_empty_axes", c -> List.of(reduce(c)));
        }
        r.add("Softmax", 1, "axis", c -> c.one(softmax(c, false)));
        r.add("LogSoftmax", 1, "axis", c -> c.one(softmax(c, true)));
        SelectionOperators.register(r);
    }

    private static OnnxValue reduce(OnnxContext c) {
        int rank = c.input(0).shape().length;
        int[] axes = c.axes(1);
        if (axes.length == 0 && c.integer("noop_with_empty_axes", 0) != 0) return c.input(0);
        if (axes.length == 0) axes = IntStream.range(0, rank).toArray();
        axes = Arrays.stream(axes).map(a -> ValueOps.axis(a, rank)).sorted().toArray();
        if (Arrays.stream(axes).distinct().count() != axes.length)
            throw new IllegalArgumentException("Duplicate reduction axis");
        boolean keep = c.integer("keepdims", 1) != 0;
        if (!c.input(0).isFloating())
            return HostReduction.reduce(c.input(0), axes, keep, c.operator());
        Tensor value = preprocess(c.tensor(0), c.operator());
        for (int i = axes.length - 1; i >= 0; i--) {
            value = reduceAxis(value, axes[i], keep, c.operator());
        }
        return OnnxValue.of(postprocess(value, c.operator()).to(c.tensor(0).dtype()));
    }

    private static Tensor preprocess(Tensor value, String operation) {
        return switch (operation) {
            case "ReduceL1" -> ArithmeticOperators.abs(value);
            case "ReduceL2", "ReduceSumSquare" -> value.multiply(value);
            default -> value;
        };
    }

    private static Tensor postprocess(Tensor value, String operation) {
        return switch (operation) {
            case "ReduceL2" -> value.sqrt();
            case "ReduceLogSum" -> value.log();
            default -> value;
        };
    }

    private static Tensor reduceAxis(Tensor value, int axis, boolean keep, String operation) {
        return switch (operation) {
            case "ReduceMean" -> value.mean(axis, keep);
            case "ReduceMax" -> value.maximum(axis, keep);
            case "ReduceMin" -> value.neg().maximum(axis, keep).neg();
            case "ReduceProd" -> product(value, axis, keep);
            case "ReduceLogSumExp" -> logSumExp(value, axis, keep);
            default -> value.sum(axis, keep);
        };
    }

    private static Tensor product(Tensor input, int axis, boolean keep) {
        int[] shape = input.shape();
        shape[axis] = 1;
        Tensor output = Tensor.ones(input.dtype(), shape);
        for (int i = 0; i < input.size(axis); i++) {
            output = output.multiply(input.slice(axis, i, i + 1));
        }
        return keep ? output : output.select(axis, 0);
    }

    static Tensor logSumExp(Tensor input, int axis, boolean keep) {
        Tensor value = input.to(io.github.kirstenali.deepj.tensor.DType.FLOAT32);
        Tensor maximum = value.maximum(axis, true).clamp(-Float.MAX_VALUE, Float.MAX_VALUE);
        Tensor output = value.subtract(maximum).exp().sum(axis, true).log().add(maximum).to(input.dtype());
        return keep ? output : output.select(axis, 0);
    }

    private static Tensor softmax(OnnxContext c, boolean logarithm) {
        Tensor value = c.tensor(0);
        int axis = ValueOps.axis(c.integer("axis", c.opset() < 13 ? 1 : -1), value.rank());
        if (c.opset() >= 13)
            return logarithm ? value.subtract(logSumExp(value, axis, true)) : value.softmax(axis);
        int rows = ValueOps.product(value.shape(), 0, axis);
        int columns = ValueOps.product(value.shape(), axis, value.rank());
        Tensor flat = value.reshape(rows, columns);
        Tensor result = logarithm ? flat.subtract(logSumExp(flat, -1, true)) : flat.softmax(-1);
        return result.reshape(value.shape());
    }
}
