package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import com.google.protobuf.ByteString;

import io.github.kirstenali.deepj.onnx.proto.Onnx;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

class TensorLoadingTest {
    @TempDir Path directory;

    @ParameterizedTest
    @ValueSource(ints = {10, 16})
    void halfPrecisionRawAndTypedWeightsPreserveRepresentableValues(int type) throws Exception {
        short bits = type == 10 ? (short) 0x3e00 : (short) 0x3fc0;
        var raw = tensor(type).setRawData(ByteString.copyFrom(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN)
                                                .putShort(bits).array())).build();
        var typed = tensor(type).addInt32Data(bits & 65535).build();
        assertEquals(1.5f, OnnxTensors.read(raw, directory, OnnxOptions.defaults()).tensor().item());
        assertEquals(1.5f, OnnxTensors.read(typed, directory, OnnxOptions.defaults()).tensor().item());
    }

    @Test
    void int64WeightsDoNotRoundThroughFloatingPoint() throws Exception {
        long value = (1L << 60) + 7;
        var tensor = tensor(7).addInt64Data(value).build();
        assertArrayEquals(new long[] {value}, OnnxTensors.read(tensor, directory, OnnxOptions.defaults()).longs());
    }

    @Test
    void refusesTruncatedAndInvalidTensorPayloads() {
        var raw = tensor(1).setRawData(ByteString.copyFrom(new byte[] {0})).build();
        assertThrows(IOException.class, () -> OnnxTensors.read(raw, directory, OnnxOptions.defaults()));
        var typed = tensor(9).addInt32Data(2).build();
        assertThrows(IOException.class, () -> OnnxTensors.read(typed, directory, OnnxOptions.defaults()));
        var missing = tensor(1).build();
        assertThrows(IOException.class, () -> OnnxTensors.read(missing, directory, OnnxOptions.defaults()));
    }

    @Test
    void externalDataCannotEscapeItsDirectoryOrUseParentSegments() throws Exception {
        Path inside = directory.resolve("weight.bin");
        Files.write(inside, new byte[4]);
        Path child = Files.createDirectory(directory.resolve("child"));
        var tensor = external("../weight.bin");
        assertThrows(IOException.class, () -> OnnxTensors.read(tensor, child, OnnxOptions.defaults()));
        Path link = child.resolve("link.bin");
        Files.createSymbolicLink(link, inside);
        assertThrows(IOException.class, () -> OnnxTensors.read(external("link.bin"), child, OnnxOptions.defaults()));
    }

    @Test
    void validatesExternalOffsetsAndLengths() throws Exception {
        Files.write(directory.resolve("weight.bin"), new byte[4]);
        var offset = external("weight.bin").toBuilder().addExternalData(entry("offset", "1")).build();
        assertThrows(IOException.class, () -> OnnxTensors.read(offset, directory, OnnxOptions.defaults()));
        var tensor = external("weight.bin").toBuilder().addExternalData(entry("length", "3")).build();
        var wrong = tensor;
        assertThrows(IOException.class, () -> OnnxTensors.read(wrong, directory, OnnxOptions.defaults()));
    }

    @Test
    void integerInputsRejectOutOfRangeValuesAndRetainExactShape() {
        assertThrows(IllegalArgumentException.class, () -> OnnxValue.integerTensor(2, new long[] {256}, 1));
        assertThrows(IllegalArgumentException.class, () -> OnnxValue.integerTensor(13, new long[] {-1}, 1));
        assertEquals(0, OnnxValue.elements(new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE, 0}));
        assertEquals(0, ValueOps.product(new int[]{Integer.MAX_VALUE, Integer.MAX_VALUE, 0}, 0, 3));
    }

    private static Onnx.TensorProto.Builder tensor(int type) {
        return Onnx.TensorProto.newBuilder().setName("value").setDataType(type).addDims(1);
    }

    private static Onnx.TensorProto external(String location) {
        return tensor(1).setDataLocation(Onnx.TensorProto.DataLocation.EXTERNAL)
                .addExternalData(entry("location", location)).build();
    }

    private static Onnx.StringStringEntryProto entry(String key, String value) {
        return Onnx.StringStringEntryProto.newBuilder().setKey(key).setValue(value).build();
    }
}
