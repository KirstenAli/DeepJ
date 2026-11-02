package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;

/** Explicit host staging at the network boundary; all model and optimizer arithmetic stays in DeepJ. */
final class TensorTransfer {
    private TensorTransfer() {}

    static void broadcast(ProcessGroup group, String key, List<Tensor> tensors, int bucketElements) {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            for (int index = 0; index < tensors.size(); index++) {
                broadcastTensor(group, key + "." + index, tensors.get(index), bucketElements);
            }
        }
    }

    private static void broadcastTensor(ProcessGroup group, String key, Tensor tensor, int bucketElements) {
        try (var scope = TensorScope.open()) {
            float[] values = group.rank() == 0 ? tensor.toFloatArray() : new float[tensor.numel()];
            for (int offset = 0; offset < values.length; offset += bucketElements) {
                int end = Math.min(values.length, offset + bucketElements);
                float[] payload = java.util.Arrays.copyOfRange(values, offset, end);
                group.broadcast(key + "." + offset, payload);
                System.arraycopy(payload, 0, values, offset, payload.length);
            }
            if (group.rank() != 0) tensor.copyFrom(Tensor.from(values, tensor.dtype(), tensor.shape()));
        }
    }
}
