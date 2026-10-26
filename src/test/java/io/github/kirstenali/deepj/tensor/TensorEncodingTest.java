package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.*;
import java.nio.ByteBuffer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorEncodingTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void everyCompactBitPatternRoundTripsInTwoBytesWithoutFloatShadows() throws IOException {
        for (DType type : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor source = new Tensor(type, 2, 128, 256), target = new Tensor(type, source.shape());
            short[] original = source.memory().shorts();
            for (int bits = 0; bits <= 0xffff; bits++) {
                original[bits] = (short) bits;
            }
            byte[] bytes = encode(source);
            TensorEncoding.readInto(input(bytes), target);
            assertArrayEquals(original, target.memory().shorts());
            assertEquals(28 + 65536 * 2, bytes.length);
            assertEquals(65536 * 2, target.storageBytes());
            assertThrows(IllegalStateException.class, () -> target.memory().rawFloats());
        }
    }

    @Test
    void float32PayloadsPreserveSignedZeroAndSignalingNanBits() throws IOException {
        int[] bits = {0x80000000, 0x7f812345, 0xffc12345, 1};
        Tensor source = new Tensor(2, 2), target = new Tensor(2, 2);
        for (int index = 0; index < bits.length; index++) {
            source.rawData()[index] = Float.intBitsToFloat(bits[index]);
        }
        TensorEncoding.readInto(input(encode(source)), target);
        for (int index = 0; index < bits.length; index++) {
            assertEquals(bits[index], Float.floatToRawIntBits(target.rawData()[index]));
        }
    }

    @Test
    void scalarEmptyAndOffsetViewsUseLogicalShapeAndValues() throws IOException {
        for (DType dtype : DType.values()) {
            Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, dtype, 3, 2).slice(0, 1, 3)
                    .transpose(0, 1);
            Tensor target = new Tensor(dtype, source.shape());
            TensorEncoding.readInto(input(encode(source)), target);
            assertArrayEquals(source.toFloatArray(), target.toFloatArray());
            assertDescriptor(new Tensor(dtype, new int[0]), 1);
            assertDescriptor(new Tensor(dtype, 2, 0, 3), 0);
        }
    }

    private static void assertDescriptor(Tensor tensor, int elements) throws IOException {
        TensorDescriptor descriptor = TensorDescriptor.read(input(encode(tensor)));
        assertArrayEquals(tensor.shape(), descriptor.shape());
        assertEquals(tensor.dtype(), descriptor.dtype());
        assertEquals(elements, descriptor.elements());
        assertEquals((long) elements * tensor.dtype().bytes(), descriptor.bytes());
        assertEquals(descriptor, new TensorDescriptor(tensor.dtype(), tensor.shape()));
        assertEquals(descriptor.hashCode(), new TensorDescriptor(tensor.dtype(), tensor.shape()).hashCode());
    }

    @Test
    void descriptorsAreDefensiveAndMalformedHeadersCannotChangeValues() throws IOException {
        Tensor target = Tensor.from(new float[]{1, 2}, 1, 2);
        TensorDescriptor descriptor = new TensorDescriptor(DType.FLOAT32, target.shape());
        descriptor.shape()[0] = 7;
        assertArrayEquals(new int[]{1, 2}, descriptor.shape());
        assertMalformed(target, 0, 9);
        assertMalformed(target, 4, -1);
        assertMalformed(target, 4, 1025);
        assertMalformed(target, 8, -1);
        assertMalformed(target, 8, Integer.MAX_VALUE);
        byte[] bytes = encode(target);
        ByteBuffer.wrap(bytes).putLong(16, -1);
        assertThrows(IOException.class, () -> TensorEncoding.readInto(input(bytes), target));
        assertArrayEquals(new float[]{1, 2}, target.toFloatArray());
        assertEquals(0, target.gradientState().version());
    }

    private static void assertMalformed(Tensor target, int offset, int value) throws IOException {
        byte[] bytes = encode(target);
        ByteBuffer.wrap(bytes).putInt(offset, value);
        assertThrows(IOException.class, () -> TensorEncoding.readInto(input(bytes), target));
    }

    @Test
    void dtypeShapeAndOverlappingDestinationsRejectBeforeWrites() throws IOException {
        byte[] bytes = encode(new Tensor(DType.FLOAT16, 2));
        assertThrows(IOException.class, () -> TensorEncoding.readInto(input(bytes), new Tensor(2)));
        assertThrows(IOException.class, () -> TensorEncoding.readInto(input(bytes), new Tensor(DType.FLOAT16, 3)));
        Tensor source = Tensor.from(new float[]{3, 4});
        Tensor overlap = Tensor.from(new float[]{1}).expand(2);
        assertThrows(IllegalStateException.class, () -> TensorEncoding.readInto(input(encode(source)), overlap));
        assertThrows(IOException.class, () -> TensorEncoding.readInto(input(new byte[0]), source));
    }

    @Test
    void rawRestorationPreservesCompactSubviewBitsAndUpdatesTheSharedVersion() {
        Tensor source = new Tensor(DType.BFLOAT16, 2), parent = new Tensor(DType.BFLOAT16, 3);
        source.memory().shorts()[0] = (short) 0x7f81;
        source.memory().shorts()[1] = (short) 0x8000;
        TensorEncoding.restore(parent.slice(0, 1, 3), source);
        assertArrayEquals(new short[]{0, (short) 0x7f81, (short) 0x8000}, parent.memory().shorts());
        assertEquals(1, parent.gradientState().version());
        assertThrows(IllegalArgumentException.class, () -> TensorEncoding.restore(new Tensor(2), source));
        assertThrows(IllegalArgumentException.class,
                () -> TensorEncoding.restore(new Tensor(DType.BFLOAT16, 3), source));
    }

    private static byte[] encode(Tensor tensor) throws IOException {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) { TensorEncoding.write(output, tensor); }
        return bytes.toByteArray();
    }

    private static DataInputStream input(byte[] bytes) { return new DataInputStream(new ByteArrayInputStream(bytes)); }
}
