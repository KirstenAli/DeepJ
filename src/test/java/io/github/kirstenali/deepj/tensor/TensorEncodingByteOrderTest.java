package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TensorEncodingByteOrderTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void bothByteOrdersPreserveRawBitsAcrossChunksAndStridedOffsets(DType dtype) throws Exception {
        int[] bits = new Random(12).ints((1 << 18) + 17).toArray();
        for (ByteOrder order : new ByteOrder[]{ByteOrder.BIG_ENDIAN, ByteOrder.LITTLE_ENDIAN}) {
            Tensor dense = new Tensor(dtype, bits.length + 2).slice(0, 1, bits.length + 1);
            Tensor strided = new Tensor(dtype, bits.length, 2).select(1, 1);
            readAndCompare(bits, dtype, order, dense);
            readAndCompare(bits, dtype, order, strided);
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void truncatedPayloadsFailAndEmptyPayloadsDoNotRead(DType dtype) throws Exception {
        var empty = new DataInputStream(new ByteArrayInputStream(new byte[0]));
        TensorEncoding.readValues(empty, new Tensor(dtype, 0), ByteOrder.LITTLE_ENDIAN);
        assertThrows(EOFException.class, () ->
                TensorEncoding.readValues(empty, new Tensor(dtype, 1), ByteOrder.LITTLE_ENDIAN));
    }

    private static void readAndCompare(int[] bits, DType dtype, ByteOrder order, Tensor target) throws Exception {
        ByteBuffer payload = ByteBuffer.allocate(bits.length * dtype.bytes()).order(order);
        for (int value : bits) {
            if (dtype == DType.FLOAT32) payload.putInt(value);
            else payload.putShort((short) value);
        }
        TensorEncoding.readValues(new DataInputStream(new ByteArrayInputStream(payload.array())), target, order);
        for (int index = 0; index < bits.length; index++) {
            int address = target.layout().addressAt(index);
            int actual = dtype == DType.FLOAT32 ? Float.floatToRawIntBits(target.memory().get(address))
                    : target.memory().shorts()[address] & 0xffff;
            assertEquals(dtype == DType.FLOAT32 ? bits[index] : bits[index] & 0xffff, actual);
        }
    }
}
