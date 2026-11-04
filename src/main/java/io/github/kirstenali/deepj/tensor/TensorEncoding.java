package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradMode;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Raw-bit, logical-order tensor payloads; no FLOAT32 shadow is needed for compact storage. */
public final class TensorEncoding {
    private static final int CHUNK = 1 << 18;
    private TensorEncoding() {}

    public static void write(DataOutput output, Tensor tensor) throws IOException {
        new TensorDescriptor(tensor.dtype(), tensor.shape()).write(output);
        writeValues(output, tensor);
    }

    public static void writeValues(DataOutput output, Tensor tensor) throws IOException {
        tensor.materialize();
        if (tensor.isContiguous()) {
            writeDense(output, tensor);
            return;
        }
        for (int index = 0; index < tensor.numel(); index++) {
            int address = tensor.layout().addressAt(index);
            if (tensor.dtype() == DType.FLOAT32) output.writeInt(Float.floatToRawIntBits(tensor.memory().get(address)));
            else output.writeShort(tensor.memory().shorts()[address]);
        }
    }

    private static void writeDense(DataOutput output, Tensor tensor) throws IOException {
        int start = tensor.storageOffset(), count = tensor.numel();
        for (int done = 0; done < count; done += CHUNK) {
            int length = Math.min(CHUNK, count - done);
            output.write(denseChunk(tensor, start + done, length), 0, length * tensor.dtype().bytes());
        }
    }

    private static byte[] denseChunk(Tensor tensor, int start, int length) {
        ByteBuffer bytes = ByteBuffer.allocate(length * tensor.dtype().bytes()).order(ByteOrder.BIG_ENDIAN);
        if (tensor.dtype() == DType.FLOAT32) bytes.asFloatBuffer().put(tensor.memory().floats(), start, length);
        else bytes.asShortBuffer().put(tensor.memory().shorts(), start, length);
        return bytes.array();
    }

    public static void readInto(DataInput input, Tensor tensor) throws IOException {
        TensorDescriptor.read(input).requireMatches(tensor);
        readValues(input, tensor);
    }

    /** Explicit checkpoint transfer; private host staging is released after an independent backend copy. */
    public static Tensor readValuesToBackend(DataInput input, DType dtype, int... shape) throws IOException {
        Tensor staging = Tensor.deviceOnly(dtype, shape);
        try {
            readValues(input, staging);
            return HostStaging.copy(staging);
        } finally {
            staging.memory().discardStaleHost();
        }
    }

    /** Restores a lazily allocated model weight without retaining its host payload on a device backend. */
    public static void readWeight(DataInput input, Tensor target) throws IOException {
        if (Tensor.backend().usesDeviceStorage() && target.coversStorage()
                && !target.hasHostStorage() && target.numel() != 0) {
            restoreWeight(target, readValues(input, target.dtype(), target.shape()), false);
        } else {
            readValues(input, target);
        }
    }

    private static Tensor readValues(DataInput input, DType dtype, int... shape) throws IOException {
        Tensor source = new Tensor(dtype, shape);
        readValues(input, source);
        return source;
    }

    /** Transfers a private, contiguous checkpoint tensor and retires its staging buffer before returning. */
    public static void restoreWeight(Tensor target, Tensor source, boolean transpose) {
        try (var ignored = GradMode.noGrad(); var buffers = BackwardBuffers.begin()) {
            try {
                buffers.capture(() -> {
                    Tensor staged = Tensor.backend().usesDeviceStorage() ? HostStaging.copy(source) : source;
                    Tensor value = transpose ? staged.transpose() : staged.reshape(target.shape());
                    return target.copyFrom(value);
                });
            } finally {
                buffers.releaseTemporaryResources();
            }
        }
    }

    public static void readValues(DataInput input, Tensor tensor) throws IOException {
        tensor.layout().requireWritable();
        tensor.materialize();
        try (var ignored = GradMode.noGrad()) {
            tensor.gradientState().beforeMutation();
            if (tensor.isContiguous()) readDense(input, tensor);
            else readStrided(input, tensor);
        } finally {
            TensorStorage.markGpuNeedsUpload(tensor);
        }
    }

    private static void readDense(DataInput input, Tensor tensor) throws IOException {
        int start = tensor.storageOffset(), end = start + tensor.numel();
        if (tensor.dtype() == DType.FLOAT32) {
            float[] values = tensor.memory().floats();
            for (int index = start; index < end; index++) {
                values[index] = Float.intBitsToFloat(input.readInt());
            }
        } else {
            short[] values = tensor.memory().shorts();
            for (int index = start; index < end; index++) {
                values[index] = input.readShort();
            }
        }
    }

    private static void readStrided(DataInput input, Tensor tensor) throws IOException {
        for (int index = 0; index < tensor.numel(); index++) {
            readValue(input, tensor, tensor.layout().addressAt(index));
        }
    }

    private static void readValue(DataInput input, Tensor tensor, int address) throws IOException {
        if (tensor.dtype() == DType.FLOAT32) tensor.memory().set(address, Float.intBitsToFloat(input.readInt()));
        else tensor.memory().shorts()[address] = input.readShort();
    }

    /** Explicit host-side restoration after file validation; not an operator fallback. */
    public static void restore(Tensor target, Tensor source) {
        Tensor.requireSameShape(target, source, "Checkpoint restore");
        if (target.dtype() != source.dtype()) throw new IllegalArgumentException("Checkpoint dtype must match");
        target.layout().requireWritable();
        try (var ignored = GradMode.noGrad()) {
            target.gradientState().beforeMutation();
            CpuNdAssignment.apply(target, source);
        } finally {
            TensorStorage.markGpuNeedsUpload(target);
        }
    }
}
