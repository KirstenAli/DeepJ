package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;

/** Model-only named framing; training tensors retain their existing independent framing. */
final class ModelTensorIO {
    private ModelTensorIO() {}

    static boolean named(List<Parameter> parameters, boolean typed) throws IOException {
        if (!typed || parameters.stream().allMatch(parameter -> parameter.name().isEmpty())) return false;
        var names = new HashSet<String>();
        for (Parameter parameter : parameters) {
            requireName(parameter.name());
            if (!names.add(parameter.name())) throw new IOException("Duplicate parameter name: " + parameter.name());
        }
        return true;
    }

    static void write(DataOutputStream output, Parameter parameter, boolean typed, boolean named) throws IOException {
        if (named) output.writeUTF(parameter.name());
        CheckpointTensorIO.write(output, parameter.value, typed);
    }

    static Entry descriptor(DataInputStream input, ModelFormat format) throws IOException {
        String name = format.named() ? input.readUTF() : "";
        if (format.named()) requireName(name);
        TensorDescriptor descriptor = CheckpointTensorIO.descriptor(input, format.typed());
        if (!format.typed() && descriptor.elements() == 0) throw new IOException("Invalid checkpoint tensor shape");
        return new Entry(name, descriptor);
    }

    private static void requireName(String name) throws IOException {
        if (name.length() > 1024 || !name.matches("[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)*")) {
            throw new IOException("Invalid checkpoint parameter name");
        }
    }

    record Entry(String name, TensorDescriptor tensor) {
        void requireMatches(Parameter parameter) throws IOException {
            if (!name.isEmpty() && !name.equals(parameter.name())) {
                throw new IOException("Parameter name mismatch: " + name);
            }
            tensor.requireMatches(parameter.value);
        }
    }
}
