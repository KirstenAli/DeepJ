package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

final class CheckpointTensors {
    private CheckpointTensors() {}

    static void write(DataOutputStream output, List<Tensor> values) throws IOException {
        output.writeInt(values.size());
        for (Tensor value : values) { TensorEncoding.write(output, value); }
    }

    static List<Tensor> read(DataInputStream input, List<Tensor> expected, boolean moments) throws IOException {
        if (input.readInt() != expected.size()) throw new IOException("Checkpoint tensor count differs");
        List<Tensor> values = new ArrayList<>();
        for (Tensor value : expected) {
            values.add(readTensor(input, moments ? DType.FLOAT32 : value.dtype(), value.shape()));
        }
        return List.copyOf(values);
    }

    private static Tensor readTensor(DataInputStream input, DType dtype, int[] shape) throws IOException {
        TensorDescriptor descriptor = TensorDescriptor.read(input);
        descriptor.requireMatches(dtype, shape);
        Tensor tensor = new Tensor(dtype, shape);
        TensorEncoding.readValues(input, tensor);
        return tensor;
    }

    static void writeMasters(DataOutputStream output, Map<Integer, Tensor> values) throws IOException {
        output.writeInt(values.size());
        for (var entry : new TreeMap<>(values).entrySet()) {
            output.writeInt(entry.getKey());
            TensorEncoding.write(output, entry.getValue());
        }
    }

    static Map<Integer, Tensor> readMasters(DataInputStream input, List<Tensor> parameters) throws IOException {
        int count = input.readInt();
        if (count < 0 || count > parameters.size()) throw new IOException("Invalid master-weight count");
        Map<Integer, Tensor> result = new HashMap<>();
        for (int entry = 0; entry < count; entry++) {
            int index = input.readInt();
            if (index < 0 || index >= parameters.size() || result.containsKey(index)) {
                throw new IOException("Invalid master-weight parameter index");
            }
            result.put(index, readTensor(input, DType.FLOAT32, parameters.get(index).shape()));
        }
        return result;
    }
}
