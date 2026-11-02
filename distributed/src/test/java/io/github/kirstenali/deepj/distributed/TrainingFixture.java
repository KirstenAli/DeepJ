package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.nn.NdSupervisedTraining;
import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import java.util.Arrays;

record TrainingFixture(TrainingObjective objective, AdamW optimizer) {
    static TrainingFixture create(String family, DType dtype, long seed, int rank, int workers) {
        ShardSampler sampler = new ShardSampler(7, rank, workers, 42);
        TrainingObjective task = DecoderCases.families().contains(family)
                ? new CausalObjective(DecoderCases.model(family, dtype, seed),
                        new CausalShard(DecoderCases.data(), sampler))
                : new TensorObjective(ModelCases.tensor(family, dtype, seed), new MSELoss(),
                        new TensorShard(ModelCases.data(family), sampler));
        return new TrainingFixture(task, AdamW.defaultAdamW(0.003f));
    }

    float referenceStep(String family, int[] indices) {
        if (objective.model() instanceof CausalLM model) {
            Batch batch = selectedTokens(indices);
            return CausalLMTraining.trainer(model, size -> batch, optimizer).trainStep(indices.length);
        }
        TensorBatch all = ModelCases.data(family);
        var trainer = NdSupervisedTraining.trainer((Layer) objective.model(), new MSELoss(), optimizer,
                all.inputs().indexSelect(0, indices), all.targets().indexSelect(0, indices), 1);
        return trainer.trainStep(indices.length);
    }

    private static Batch selectedTokens(int[] indices) {
        Batch all = DecoderCases.data();
        int[][] inputs = Arrays.stream(indices).mapToObj(index -> all.x()[index]).toArray(int[][]::new);
        int[][] targets = Arrays.stream(indices).mapToObj(index -> all.y()[index]).toArray(int[][]::new);
        boolean[][] masks = Arrays.stream(indices).mapToObj(index -> all.mask(index)).toArray(boolean[][]::new);
        return new Batch(inputs, targets, masks);
    }
}
