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

    /** Consumes private checkpoint staging, uploading once and retiring temporary dtype-conversion storage. */
    public static Tensor weightToBackend(Tensor source, DType dtype) {
        java.util.Objects.requireNonNull(dtype, "dtype");
        try (var ignored = GradMode.noGrad(); var buffers = BackwardBuffers.begin()) {
            Tensor result = buffers.capture(() -> (Tensor.backend().usesDeviceStorage()
                    ? HostStaging.copy(source) : source).to(dtype));
            buffers.expose(result);
            return result;
        }
    }

    public static void readValues(DataInput input, Tensor tensor) throws IOException {
        readValues(input, tensor, ByteOrder.BIG_ENDIAN);
    }

    /** Reads a raw payload in its file byte order without creating a FLOAT32 compact shadow. */
    public static void readValues(DataInput input, Tensor tensor, ByteOrder order) throws IOException {
        java.util.Objects.requireNonNull(order, "order");
        tensor.layout().requireWritable();
        tensor.materialize();
        try (var ignored = GradMode.noGrad()) {
            tensor.gradientState().beforeMutation();
            if (tensor.isContiguous()) readDense(input, tensor, order);
            else readStrided(input, tensor, order);
        } finally {
            TensorStorage.markGpuNeedsUpload(tensor);
        }
    }

    private static void readDense(DataInput input, Tensor tensor, ByteOrder order) throws IOException {
        int capacity = Math.min(CHUNK, tensor.numel());
        ByteBuffer bytes = ByteBuffer.allocate(capacity * tensor.dtype().bytes()).order(order);
        for (int done = 0; done < tensor.numel(); done += capacity) {
            int count = Math.min(capacity, tensor.numel() - done);
            input.readFully(bytes.array(), 0, count * tensor.dtype().bytes());
            readDenseChunk(tensor, bytes, tensor.storageOffset() + done, count);
        }
    }

    private static void readDenseChunk(Tensor tensor, ByteBuffer bytes, int start, int count) {
        if (tensor.dtype() == DType.FLOAT32) bytes.asFloatBuffer().get(tensor.memory().floats(), start, count);
        else bytes.asShortBuffer().get(tensor.memory().shorts(), start, count);
    }

    private static void readStrided(DataInput input, Tensor tensor, ByteOrder order) throws IOException {
        for (int index = 0; index < tensor.numel(); index++) {
            readValue(input, tensor, tensor.layout().addressAt(index), order == ByteOrder.LITTLE_ENDIAN);
        }
    }

    private static void readValue(DataInput input, Tensor tensor, int address, boolean little) throws IOException {
        if (tensor.dtype() == DType.FLOAT32) {
            int bits = input.readInt();
            tensor.memory().set(address, Float.intBitsToFloat(little ? Integer.reverseBytes(bits) : bits));
        } else {
            short bits = input.readShort();
            tensor.memory().shorts()[address] = little ? Short.reverseBytes(bits) : bits;
        }
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
