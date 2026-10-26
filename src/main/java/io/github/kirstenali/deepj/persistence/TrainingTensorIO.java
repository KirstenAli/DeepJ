package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class TrainingTensorIO {
    private TrainingTensorIO() {}

    static boolean requiresTyped(List<Parameter> parameters, AdamW.State optimizer) {
        return !optimizer.masterWeights().isEmpty()
                || parameters.stream().anyMatch(parameter -> CheckpointTensorIO.requiresTyped(parameter.value));
    }

    static void write(DataOutputStream output, List<Parameter> params, AdamW.State optimizer,
                        boolean typed) throws IOException {
        output.writeInt(params.size());
        for (int index = 0; index < params.size(); index++) {
            CheckpointTensorIO.write(output, params.get(index).value, typed);
            CheckpointTensorIO.write(output, optimizer.firstMoments().get(index), typed);
            CheckpointTensorIO.write(output, optimizer.secondMoments().get(index), typed);
            if (typed) writeMaster(output, optimizer.masterWeights().get(index));
        }
    }

    private static void writeMaster(DataOutputStream output, Tensor master) throws IOException {
        output.writeBoolean(master != null);
        if (master != null) CheckpointTensorIO.write(output, master, true);
    }

    static State read(CheckpointInput input, List<Parameter> params,
                        boolean typed) throws IOException {
        var result = new State(new ArrayList<>(params.size()), new ArrayList<>(params.size()),
                new ArrayList<>(params.size()), new LinkedHashMap<>());
        for (int index = 0; index < params.size(); index++) {
            Tensor template = params.get(index).value;
            result.model().add(CheckpointTensorIO.read(input, template.dtype(), template.shape(), typed));
            result.first().add(CheckpointTensorIO.read(input, DType.FLOAT32, template.shape(), typed));
            result.second().add(CheckpointTensorIO.read(input, DType.FLOAT32, template.shape(), typed));
            if (typed) readMaster(input, template, index, result.masters());
        }
        return result;
    }

    private static void readMaster(CheckpointInput input, Tensor template, int index,
                                    Map<Integer, Tensor> masters) throws IOException {
        boolean present = CheckpointTensorIO.readFlag(input.data()), expected = template.dtype() != DType.FLOAT32;
        if (present != expected) throw new IOException("Checkpoint master entry does not match parameter dtype");
        if (present) masters.put(index,
                CheckpointTensorIO.read(input, DType.FLOAT32, template.shape(), true));
    }

    static void apply(List<Parameter> params, List<Tensor> values) {
        for (int index = 0; index < params.size(); index++) {
            TensorEncoding.restore(params.get(index).value, values.get(index));
        }
    }

    record State(List<Tensor> model, List<Tensor> first, List<Tensor> second, Map<Integer, Tensor> masters) {}
}
