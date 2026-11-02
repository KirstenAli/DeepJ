package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;

record CheckpointImage(byte[] cohort, DistributedStep progress, List<Tensor> parameters, List<Tensor> buffers,
                       AdamW.State optimizer, byte[] workerState) {}
