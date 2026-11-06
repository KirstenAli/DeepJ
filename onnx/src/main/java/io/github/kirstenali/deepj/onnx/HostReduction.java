package io.github.kirstenali.deepj.onnx;

import java.util.Arrays;
import java.util.stream.IntStream;

final class HostReduction {
    private HostReduction() {}

    static OnnxValue reduce(OnnxValue input, int[] axes, boolean keep, String operation) {
        int[] shape = input.shape();
        int[] outputShape = keep ? IntStream.range(0, shape.length)
                                .map(i -> Arrays.binarySearch(axes, i) >= 0 ? 1 : shape[i]).toArray()
                        : IntStream.range(0, shape.length).filter(i -> Arrays.binarySearch(axes, i) < 0)
                                .map(i -> shape[i]).toArray();
        long[] output = new long[OnnxValue.elements(outputShape)];
        Arrays.fill(output, initial(operation));
        for (int i = 0; i < input.size(); i++) {
            int index = address(i, shape, axes);
            output[index] = combine(output[index], input.data()[i], operation);
        }
        return OnnxValue.host(output, outputShape, input.elementType());
    }

    private static int address(int linear, int[] shape, int[] axes) {
        int address = 0, stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            int coordinate = linear % shape[i];
            linear /= shape[i];
            if (Arrays.binarySearch(axes, i) >= 0) continue;
            address += coordinate * stride;
            stride *= shape[i];
        }
        return address;
    }

    private static long initial(String operation) {
        return switch (operation) {
            case "ReduceProd" -> 1;
            case "ReduceMax" -> Long.MIN_VALUE;
            case "ReduceMin" -> Long.MAX_VALUE;
            case "ReduceSum", "ReduceL1", "ReduceSumSquare" -> 0;
            default -> throw new IllegalArgumentException("Integer reduction unsupported: " + operation);
        };
    }

    private static long combine(long accumulator, long value, String operation) {
        return switch (operation) {
            case "ReduceProd" -> Math.multiplyExact(accumulator, value);
            case "ReduceMax" -> Math.max(accumulator, value);
            case "ReduceMin" -> Math.min(accumulator, value);
            case "ReduceL1" -> Math.addExact(accumulator, Math.absExact(value));
            case "ReduceSumSquare" -> Math.addExact(accumulator, Math.multiplyExact(value, value));
            default -> Math.addExact(accumulator, value);
        };
    }
}
