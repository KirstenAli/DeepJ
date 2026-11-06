package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

/** Non-differentiable predicates use exact host values; selected floating data stays in DeepJ. */
final class BooleanOperators {
    private BooleanOperators() {}

    static void register(OnnxRegistry r) {
        for (String name : List.of("Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual")) {
            r.add(name, name.endsWith("OrEqual") ? 12 : 7, "", c -> List.of(compare(c.input(0), c.input(1), name)));
        }
        for (String name : List.of("And", "Or", "Xor")) {
            r.add(name, 7, "", c -> List.of(logical(c.input(0), c.input(1), name)));
        }
        r.add("Not", 1, "", c -> List.of(not(c.input(0))));
        r.add("Where", 9, "", c -> List.of(where(c.input(0), c.input(1), c.input(2))));
    }

    static OnnxValue compare(Tensor a, Tensor b, String operation) {
        return compare(OnnxValue.of(a), OnnxValue.of(b), operation);
    }

    private static OnnxValue compare(OnnxValue a, OnnxValue b, String operation) {
        ValueOps.sameType(a, b);
        int[] shape = ValueOps.broadcast(a.shape(), b.shape());
        float[] first = a.isFloating() ? a.tensor().toFloatArray() : null;
        float[] second = b.isFloating() ? b.tensor().toFloatArray() : null;
        boolean[] output = new boolean[OnnxValue.elements(shape)];
        for (int i = 0; i < output.length; i++) {
            int x = ValueOps.address(i, shape, a.shape()), y = ValueOps.address(i, shape, b.shape());
            output[i] = first == null ? compare(a.data()[x], b.data()[y], operation)
                            : compare(first[x], second[y], operation);
        }
        return OnnxValue.booleans(output, shape);
    }

    private static boolean compare(long a, long b, String operation) {
        return switch (operation) {
            case "Equal" -> a == b;
            case "Greater" -> a > b;
            case "Less" -> a < b;
            case "GreaterOrEqual" -> a >= b;
            default -> a <= b;
        };
    }

    private static boolean compare(float a, float b, String operation) {
        return switch (operation) {
            case "Equal" -> a == b;
            case "Greater" -> a > b;
            case "Less" -> a < b;
            case "GreaterOrEqual" -> a >= b;
            default -> a <= b;
        };
    }

    private static OnnxValue logical(OnnxValue a, OnnxValue b, String name) {
        requireBool(a);
        requireBool(b);
        return ValueOps.binary(a, b, (x, y) -> switch (name) {
                            case "And" -> x & y;
                            case "Or" -> x | y;
                            default -> x ^ y;
                        });
    }

    private static OnnxValue not(OnnxValue value) {
        requireBool(value);
        long[] output = value.longs();
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 - output[i];
        }
        return OnnxValue.host(output, value.shape(), 9);
    }

    static Tensor where(OnnxValue condition, Tensor a, Tensor b) {
        return where(condition, OnnxValue.of(a), OnnxValue.of(b)).tensor();
    }

    private static OnnxValue where(OnnxValue condition, OnnxValue a, OnnxValue b) {
        requireBool(condition);
        ValueOps.sameType(a, b);
        int[] shape = ValueOps.broadcast(ValueOps.broadcast(a.shape(), b.shape()), condition.shape());
        OnnxValue joined = ValueOps.concatenate(List.of(ValueOps.reshape(ValueOps.expand(a, shape),
                                        new int[] {OnnxValue.elements(shape)}), ValueOps.reshape(
                                        ValueOps.expand(b, shape), new int[] {OnnxValue.elements(shape)})), 0);
        int[] indices = new int[OnnxValue.elements(shape)];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i
                            + (condition.data()[ValueOps.address(i, shape, condition.shape())] == 0 ? indices.length
                                    : 0);
        }
        return ValueOps.take(joined, indices, shape);
    }

    private static void requireBool(OnnxValue value) {
        if (value.elementType() != 9)
            throw new IllegalArgumentException("ONNX condition must be BOOL");
    }
}
