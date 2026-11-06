package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;

final class CastOperators {
    private CastOperators() {}

    static void register(OnnxRegistry registry) {
        registry.add(
                "Cast",
                9,
                "to saturate round_mode",
                context -> List.of(cast(context.input(0), context.integer("to", 0))));
        registry.add(
                "CastLike",
                15,
                "saturate round_mode",
                context -> List.of(cast(context.input(0), context.input(1).elementType())));
        registry.add("Range", 11, "", context -> List.of(range(context)));
    }

    static OnnxValue cast(OnnxValue input, int type) {
        if (!OnnxTensors.supported(type))
            throw new IllegalArgumentException("Unsupported ONNX Cast target " + type);
        if (type == input.elementType()) return input;
        if (OnnxTensors.floating(type)) return toFloating(input, type);
        long[] values =
                input.isFloating()
                        ? integerValues(input.tensor(), type)
                        : convert(input.data(), type);
        return OnnxValue.host(values, input.shape(), type);
    }

    private static OnnxValue toFloating(OnnxValue input, int type) {
        DType dtype = type == 1 ? DType.FLOAT32 : type == 10 ? DType.FLOAT16 : DType.BFLOAT16;
        if (input.isFloating()) return OnnxValue.of(input.tensor().to(dtype));
        float[] data = new float[input.size()];
        for (int i = 0; i < data.length; i++) {
            data[i] = input.data()[i];
        }
        return OnnxValue.of(Tensor.from(data, dtype, input.shape()));
    }

    private static long[] convert(long[] values, int type) {
        long[] output = values.clone();
        for (int i = 0; i < output.length; i++) {
            output[i] = narrow(output[i], type);
        }
        return output;
    }

    private static long narrow(long value, int type) {
        return switch (type) {
            case 2 -> value & 255;
            case 3 -> (byte) value;
            case 4 -> value & 65535;
            case 5 -> (short) value;
            default -> wide(value, type);
        };
    }

    private static long wide(long value, int type) {
        return switch (type) {
            case 6 -> (int) value;
            case 9 -> value == 0 ? 0 : 1;
            case 12 -> value & 0xffff_ffffL;
            default -> value;
        };
    }

    private static OnnxValue range(OnnxContext context) {
        ValueOps.sameType(context.input(0), context.input(1));
        ValueOps.sameType(context.input(0), context.input(2));
        if (!context.input(0).isFloating()) return integerRange(context);
        float start = context.tensor(0).item(),
                stop = context.tensor(1).item(),
                step = context.tensor(2).item();
        if (step == 0 || !Float.isFinite(start + stop + step))
            throw new IllegalArgumentException("Invalid Range bounds");
        double length = Math.max(0, Math.ceil(((double) stop - start) / step));
        if (length > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Range exceeds tensor element limit");
        float[] output = new float[(int) length];
        for (int i = 0; i < output.length; i++) {
            output[i] = start + i * step;
        }
        return OnnxValue.of(Tensor.from(output, context.tensor(0).dtype(), output.length));
    }

    private static OnnxValue integerRange(OnnxContext context) {
        long start = context.input(0).scalar(),
                stop = context.input(1).scalar(),
                step = context.input(2).scalar();
        if (step == 0) throw new IllegalArgumentException("Range step cannot be zero");
        long distance = Math.subtractExact(stop, start);
        long count =
                distance == 0 || Long.signum(distance) != Long.signum(step)
                        ? 0
                        : Math.addExact(
                                Math.floorDiv(
                                        Math.subtractExact(Math.abs(distance), 1), Math.abs(step)),
                                1);
        long[] output = new long[Math.toIntExact(count)];
        for (int i = 0; i < output.length; i++) {
            output[i] = Math.addExact(start, Math.multiplyExact((long) i, step));
        }
        return OnnxValue.host(output, new int[] {output.length}, context.input(0).elementType());
    }

    private static long[] integerValues(Tensor input, int type) {
        float[] values = input.toFloatArray();
        long[] output = new long[values.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = integer(values[i], type);
        }
        return output;
    }

    private static long integer(float value, int type) {
        if (type == 9) return value == 0 ? 0 : 1;
        if (!Float.isFinite(value) || value < Long.MIN_VALUE || value >= 0x1p63f) {
            throw new IllegalArgumentException(
                    "Floating value is outside the exact supported integer range");
        }
        return narrow((long) value, type);
    }
}
