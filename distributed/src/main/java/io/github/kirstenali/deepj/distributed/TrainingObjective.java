package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.nn.Module;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;
import java.util.List;

/** Opt-in training adapter; the underlying model, tensor backend, and autograd remain ordinary DeepJ objects. */
public interface TrainingObjective {
    Trainable model();
    WeightedLoss evaluate(int batchSize);
    default float gradientClipNorm() { return Float.POSITIVE_INFINITY; }
    default WorkerState workerState() { return WorkerState.empty(); }
    default List<Tensor> buffers() { return model() instanceof Module module ? module.buffers() : List.of(); }
}
