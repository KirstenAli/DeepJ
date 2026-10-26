package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.tensor.*;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/** Shared version-aware tensor framing for model and resumable-training files. */
final class CheckpointTensorIO {
    private CheckpointTensorIO() {}

    static boolean requiresTyped(Tensor tensor) {
        return tensor.rank() != 2 || tensor.dtype() != DType.FLOAT32 || tensor.numel() == 0;
    }

    static void write(DataOutputStream output, Tensor tensor, boolean typed) throws IOException {
        if (typed) TensorEncoding.write(output, tensor);
        else {
            if (tensor.dtype() != DType.FLOAT32 || tensor.rank() != 2) {
                throw new IOException("Matrix checkpoint payload requires rank-two FLOAT32 storage");
            }
            output.writeInt(tensor.matrixRows());
            output.writeInt(tensor.matrixCols());
            TensorEncoding.writeValues(output, tensor);
        }
    }

    static TensorDescriptor descriptor(DataInputStream input, boolean typed) throws IOException {
        if (typed) return TensorDescriptor.read(input);
        int rows = input.readInt(), columns = input.readInt();
        try {
            return new TensorDescriptor(DType.FLOAT32, new int[]{rows, columns});
        } catch (IllegalArgumentException | ArithmeticException error) {
            throw new IOException("Invalid checkpoint tensor shape", error);
        }
    }

    static TensorDescriptor inspect(CheckpointInput input, boolean typed) throws IOException {
        TensorDescriptor descriptor = descriptor(input.data(), typed);
        if (!typed && descriptor.elements() == 0) throw new IOException("Invalid checkpoint tensor shape");
        input.skip(descriptor.bytes());
        return descriptor;
    }

    static Tensor read(CheckpointInput input, DType type, int[] shape,
                        boolean typed) throws IOException {
        TensorDescriptor descriptor = descriptor(input.data(), typed);
        descriptor.requireMatches(type, shape);
        input.requireRemaining(descriptor.bytes());
        Tensor tensor = new Tensor(type, shape);
        TensorEncoding.readValues(input.data(), tensor);
        return tensor;
    }

    static void readInto(DataInputStream input, Tensor tensor, boolean typed) throws IOException {
        descriptor(input, typed).requireMatches(tensor);
        TensorEncoding.readValues(input, tensor);
    }

    static boolean readFlag(DataInputStream input) throws IOException {
        int value = input.readUnsignedByte();
        if (value > 1) throw new IOException("Invalid checkpoint boolean flag");
        return value == 1;
    }
}
