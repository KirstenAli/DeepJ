package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

final class TrainingReport {
    private TrainingReport() {}

    static byte[] encode(TrainingFixture fixture, float loss) {
        return Wire.encode(output -> {
            output.writeFloat(loss);
            output.writeLong(fixture.optimizer().stepCount());
            for (var parameter : fixture.objective().model().parameters()) {
                output.writeUTF(parameter.name());
                TensorEncoding.write(output, parameter.value);
                TensorEncoding.write(output, parameter.grad());
            }
            var state = fixture.optimizer().state(fixture.objective().model().parameters());
            tensors(output, state.firstMoments());
            tensors(output, state.secondMoments());
            CheckpointTensors.writeMasters(output, state.masterWeights());
            tensors(output, fixture.objective().buffers());
        });
    }

    private static void tensors(DataOutputStream output, List<Tensor> tensors) throws IOException {
        output.writeInt(tensors.size());
        for (Tensor tensor : tensors) { TensorEncoding.write(output, tensor); }
    }

    static void compare(byte[] expected, byte[] actual, TrainingFixture fixture, float tolerance) throws IOException {
        try (var first = Wire.input(expected); var second = Wire.input(actual)) {
            assertEquals(first.readFloat(), second.readFloat(), tolerance);
            assertEquals(first.readLong(), second.readLong());
            for (var ignored : fixture.objective().model().parameters()) {
                String name = first.readUTF();
                assertEquals(name, second.readUTF());
                tensor(first, second, tolerance, name + " value");
                tensor(first, second, tolerance, name + " gradient");
            }
            lists(first, second, tolerance);
            lists(first, second, tolerance);
            masters(first, second, tolerance);
            lists(first, second, tolerance);
            assertEquals(-1, first.read());
            assertEquals(-1, second.read());
        }
    }

    private static void masters(DataInputStream first, DataInputStream second, float tolerance) throws IOException {
        int count = first.readInt();
        assertEquals(count, second.readInt());
        for (int index = 0; index < count; index++) {
            assertEquals(first.readInt(), second.readInt());
            tensor(first, second, tolerance, "master " + index);
        }
    }

    private static void lists(DataInputStream first, DataInputStream second, float tolerance) throws IOException {
        int count = first.readInt();
        assertEquals(count, second.readInt());
        for (int index = 0; index < count; index++) { tensor(first, second, tolerance, "moment " + index); }
    }

    private static void tensor(DataInputStream first, DataInputStream second, float tolerance, String label)
            throws IOException {
        TensorDescriptor descriptor = TensorDescriptor.read(first);
        assertEquals(descriptor, TensorDescriptor.read(second));
        Tensor a = new Tensor(descriptor.dtype(), descriptor.shape());
        Tensor b = new Tensor(descriptor.dtype(), descriptor.shape());
        TensorEncoding.readValues(first, a);
        TensorEncoding.readValues(second, b);
        assertArrayEquals(a.toFloatArray(), b.toFloatArray(), tolerance, label);
    }
}
