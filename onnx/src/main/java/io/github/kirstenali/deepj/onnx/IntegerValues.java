package io.github.kirstenali.deepj.onnx;

final class IntegerValues {
    private IntegerValues() {}

    static void validate(long[] values, int type) {
        if (!OnnxTensors.supported(type) || OnnxTensors.floating(type)) {
            throw new IllegalArgumentException("Expected supported integer/boolean element type");
        }
        long minimum = minimum(type), maximum = maximum(type);
        for (long value : values) {
            if (value < minimum || value > maximum) {
                throw new IllegalArgumentException(
                        "Value is outside supported ONNX integer type " + type);
            }
        }
    }

    private static long minimum(int type) {
        return switch (type) {
            case 2, 4, 9, 12, 13 -> 0;
            case 3 -> Byte.MIN_VALUE;
            case 5 -> Short.MIN_VALUE;
            case 6 -> Integer.MIN_VALUE;
            default -> Long.MIN_VALUE;
        };
    }

    private static long maximum(int type) {
        return switch (type) {
            case 2 -> 255;
            case 3 -> Byte.MAX_VALUE;
            case 4 -> 65535;
            case 5 -> Short.MAX_VALUE;
            default -> wideMaximum(type);
        };
    }

    private static long wideMaximum(int type) {
        return switch (type) {
            case 6 -> Integer.MAX_VALUE;
            case 9 -> 1;
            case 12 -> 0xffff_ffffL;
            default -> Long.MAX_VALUE;
        };
    }
}
