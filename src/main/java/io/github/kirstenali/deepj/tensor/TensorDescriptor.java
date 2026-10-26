package io.github.kirstenali.deepj.tensor;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/** Validated logical checkpoint metadata; reading it never allocates tensor values. */
public record TensorDescriptor(DType dtype, int[] shape) {
    private static final int MAX_CHECKPOINT_RANK = 1024;

    public TensorDescriptor {
        Objects.requireNonNull(dtype, "dtype");
        var dimensions = new TensorShape(shape);
        dimensions.denseStrides();
        if (dimensions.rank() > MAX_CHECKPOINT_RANK) {
            throw new IllegalArgumentException("Checkpoint tensor rank exceeds 1024");
        }
        shape = dimensions.dimensions();
    }

    @Override public int[] shape() { return shape.clone(); }
    public int elements() { return new TensorShape(shape).elements(); }
    public long bytes() { return dtype.storageBytes(elements()); }

    public void requireMatches(Tensor tensor) throws IOException {
        requireMatches(tensor.dtype(), tensor.shape());
    }

    public void requireMatches(DType expectedType, int[] expectedShape) throws IOException {
        if (!Arrays.equals(shape, expectedShape)) throw new IOException("Shape mismatch in checkpoint tensor");
        if (dtype != expectedType) throw new IOException("Dtype mismatch in checkpoint tensor");
    }

    public void write(DataOutput output) throws IOException {
        output.writeInt(code(dtype));
        output.writeInt(shape.length);
        for (int dimension : shape) {
            output.writeInt(dimension);
        }
        output.writeLong(bytes());
    }

    public static TensorDescriptor read(DataInput input) throws IOException {
        DType dtype = dtype(input.readInt());
        int rank = input.readInt();
        if (rank < 0 || rank > MAX_CHECKPOINT_RANK) throw new IOException("Invalid checkpoint tensor rank");
        int[] dimensions = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            dimensions[axis] = input.readInt();
        }
        TensorDescriptor descriptor = checked(dtype, dimensions);
        if (input.readLong() != descriptor.bytes()) throw new IOException("Invalid checkpoint tensor byte length");
        return descriptor;
    }

    private static TensorDescriptor checked(DType dtype, int[] shape) throws IOException {
        try {
            return new TensorDescriptor(dtype, shape);
        } catch (IllegalArgumentException | ArithmeticException error) {
            throw new IOException("Invalid checkpoint tensor shape", error);
        }
    }

    private static int code(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> 1;
            case FLOAT16 -> 2;
            case BFLOAT16 -> 3;
        };
    }

    private static DType dtype(int code) throws IOException {
        return switch (code) {
            case 1 -> DType.FLOAT32;
            case 2 -> DType.FLOAT16;
            case 3 -> DType.BFLOAT16;
            default -> throw new IOException("Unknown checkpoint tensor dtype: " + code);
        };
    }

    @Override public boolean equals(Object other) {
        return other instanceof TensorDescriptor descriptor && dtype == descriptor.dtype
                && Arrays.equals(shape, descriptor.shape);
    }

    @Override public int hashCode() { return 31 * dtype.hashCode() + Arrays.hashCode(shape); }
}
