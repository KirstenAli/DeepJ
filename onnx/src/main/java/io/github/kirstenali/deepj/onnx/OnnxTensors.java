package io.github.kirstenali.deepj.onnx;

import io.github.kirstenali.deepj.onnx.proto.Onnx;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

final class OnnxTensors {
    private OnnxTensors() {}

    static boolean supported(int type) {
        return type >= 1 && type <= 7
                || type == 9
                || type == 10
                || type == 12
                || type == 13
                || type == 16;
    }

    static OnnxValue read(Onnx.TensorProto tensor, Path directory, OnnxOptions options)
            throws IOException {
        int[] shape = tensor.getDimsList().stream().mapToInt(Math::toIntExact).toArray();
        int count = OnnxValue.elements(shape), type = tensor.getDataType();
        if (!supported(type)) throw new IOException("Unsupported ONNX tensor type " + type);
        long length = Math.multiplyExact((long) count, width(type));
        if (length > options.maxWeightBytes())
            throw new IOException("ONNX tensor exceeds weight budget");
        ByteBuffer raw = payload(tensor, directory, length);
        return floating(type)
                ? floating(tensor, raw, shape, count)
                : integer(tensor, raw, shape, count);
    }

    static boolean floating(int type) {
        return type == 1 || type == 10 || type == 16;
    }

    private static int width(int type) {
        return switch (type) {
            case 1, 6, 12 -> 4;
            case 4, 5, 10, 16 -> 2;
            case 7, 13 -> 8;
            default -> 1;
        };
    }

    private static ByteBuffer payload(Onnx.TensorProto tensor, Path directory, long length)
            throws IOException {
        if (tensor.getDataLocation() == Onnx.TensorProto.DataLocation.EXTERNAL) {
            return ExternalTensorData.read(tensor, directory, length);
        }
        if (tensor.getRawData().isEmpty()) return null;
        if (tensor.getRawData().size() != length)
            throw new IOException("ONNX tensor byte length differs");
        return tensor.getRawData().asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
    }

    private static OnnxValue floating(
            Onnx.TensorProto tensor, ByteBuffer raw, int[] shape, int count) throws IOException {
        int type = tensor.getDataType();
        if (raw == null)
            requireCount(
                    type == 1 ? tensor.getFloatDataCount() : tensor.getInt32DataCount(), count);
        float[] data = new float[count];
        for (int i = 0; i < count; i++) {
            data[i] = floatingAt(tensor, raw, i);
        }
        DType dtype = type == 1 ? DType.FLOAT32 : type == 10 ? DType.FLOAT16 : DType.BFLOAT16;
        Tensor source = Tensor.from(data, dtype, shape);
        Tensor value =
                Tensor.backend().usesDeviceStorage() ? Tensor.backend().copy(source) : source;
        return OnnxValue.of(value.requiresGrad(false).retainDeviceBuffer());
    }

    private static float floatingAt(Onnx.TensorProto tensor, ByteBuffer raw, int i) {
        if (tensor.getDataType() == 1) return raw == null ? tensor.getFloatData(i) : raw.getFloat();
        short bits = raw == null ? (short) tensor.getInt32Data(i) : raw.getShort();
        return tensor.getDataType() == 10
                ? Float.float16ToFloat(bits)
                : Float.intBitsToFloat((bits & 65535) << 16);
    }

    private static OnnxValue integer(
            Onnx.TensorProto tensor, ByteBuffer raw, int[] shape, int count) throws IOException {
        if (raw == null) requireCount(typedCount(tensor), count);
        long[] data = new long[count];
        for (int i = 0; i < count; i++) {
            data[i] = integerAt(tensor, raw, i);
        }
        if (tensor.getDataType() == 9) {
            for (long value : data) {
                if (value != 0 && value != 1) throw new IOException("Invalid ONNX boolean");
            }
        }
        return OnnxValue.host(data, shape, tensor.getDataType());
    }

    private static int typedCount(Onnx.TensorProto tensor) {
        return switch (tensor.getDataType()) {
            case 7 -> tensor.getInt64DataCount();
            case 12, 13 -> tensor.getUint64DataCount();
            default -> tensor.getInt32DataCount();
        };
    }

    private static long integerAt(Onnx.TensorProto tensor, ByteBuffer raw, int i)
            throws IOException {
        int type = tensor.getDataType();
        long value = raw == null ? typedInteger(tensor, i) : rawInteger(raw, type);
        if (type == 13 && value < 0)
            throw new IOException("UINT64 values above signed-long range are unsupported");
        return value;
    }

    private static long typedInteger(Onnx.TensorProto tensor, int i) {
        return switch (tensor.getDataType()) {
            case 7 -> tensor.getInt64Data(i);
            case 12, 13 -> tensor.getUint64Data(i);
            default -> tensor.getInt32Data(i);
        };
    }

    private static long rawInteger(ByteBuffer raw, int type) {
        return switch (type) {
            case 2, 9 -> raw.get() & 255;
            case 3 -> raw.get();
            case 4 -> raw.getShort() & 65535;
            case 5 -> raw.getShort();
            case 6 -> raw.getInt();
            case 12 -> Integer.toUnsignedLong(raw.getInt());
            default -> raw.getLong();
        };
    }

    private static void requireCount(int actual, int expected) throws IOException {
        if (actual != expected)
            throw new IOException("ONNX tensor element count differs from shape");
    }
}
