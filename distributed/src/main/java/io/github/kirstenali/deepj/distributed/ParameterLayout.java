package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

final class ParameterLayout {
    final List<Parameter> parameters;
    final List<Tensor> buffers;
    private final List<Boolean> trainable;

    ParameterLayout(TrainingObjective objective) {
        parameters = unique(objective.model().parameters());
        buffers = uniqueTensors(objective.buffers());
        trainable = parameters.stream().map(parameter -> parameter.value.requiresGrad()).toList();
    }

    static List<Parameter> unique(List<Parameter> source) {
        var seen = new IdentityHashMap<Tensor, Boolean>();
        List<Parameter> result = new ArrayList<>();
        for (Parameter parameter : source) {
            if (seen.put(parameter.value, Boolean.TRUE) == null) result.add(parameter);
        }
        return List.copyOf(result);
    }

    private static List<Tensor> uniqueTensors(List<Tensor> source) {
        var seen = new IdentityHashMap<Tensor, Boolean>();
        List<Tensor> result = new ArrayList<>();
        for (Tensor tensor : source) {
            if (seen.put(tensor, Boolean.TRUE) == null) result.add(tensor);
        }
        return List.copyOf(result);
    }

    byte[] manifest(TrainingObjective task, ParameterOptimizer optimizer, DistributedOptions options) {
        return Wire.encode(output -> {
            output.writeUTF(task.model().getClass().getName());
            output.writeUTF(optimizer.getClass().getName());
            output.writeInt(options.bucketElements());
            output.writeUTF(options.bufferPolicy().name());
            output.writeFloat(task.gradientClipNorm());
            writeParameters(output);
            writeTensors(output, buffers);
            if (optimizer instanceof AdamW adam) writeOptimizer(output, adam.state(List.of()));
        });
    }

    void verify(TrainingObjective task) {
        List<Parameter> current = unique(task.model().parameters());
        if (current.size() != parameters.size()) throw new IllegalStateException("Parameter inventory changed");
        for (int index = 0; index < current.size(); index++) {
            if (current.get(index) != parameters.get(index)
                    || current.get(index).value.requiresGrad() != trainable.get(index)) {
                throw new IllegalStateException("Parameter identities or order changed");
            }
        }
        List<Tensor> currentBuffers = uniqueTensors(task.buffers());
        if (currentBuffers.size() != buffers.size()) throw new IllegalStateException("Buffer inventory changed");
        for (int index = 0; index < buffers.size(); index++) {
            if (currentBuffers.get(index) != buffers.get(index)) {
                throw new IllegalStateException("Buffer identity changed");
            }
        }
    }

    private void writeParameters(DataOutputStream output) throws IOException {
        output.writeInt(parameters.size());
        for (Parameter parameter : parameters) {
            output.writeUTF(parameter.name());
            output.writeBoolean(parameter.value.requiresGrad());
            descriptor(parameter.value).write(output);
        }
    }

    private static void writeTensors(DataOutputStream output, List<Tensor> tensors) throws IOException {
        output.writeInt(tensors.size());
        for (Tensor tensor : tensors) { descriptor(tensor).write(output); }
    }

    static TensorDescriptor descriptor(Tensor tensor) { return new TensorDescriptor(tensor.dtype(), tensor.shape()); }

    private static void writeOptimizer(DataOutputStream output, AdamW.State state) throws IOException {
        if (state.step() != 0) {
            throw new IllegalArgumentException("Use distributed checkpoint restoration to resume AdamW");
        }
        if (!Float.isFinite(state.learningRate()) || !Float.isFinite(state.beta1()) || !Float.isFinite(state.beta2())
                || !Float.isFinite(state.epsilon()) || !Float.isFinite(state.weightDecay())) {
            throw new IllegalArgumentException("AdamW settings must be finite");
        }
        output.writeFloat(state.learningRate());
        output.writeFloat(state.beta1());
        output.writeFloat(state.beta2());
        output.writeFloat(state.epsilon());
        output.writeFloat(state.weightDecay());
    }
}
