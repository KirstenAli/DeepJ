package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
import java.io.IOException;

record StepTotals(long weight, float meanLoss) {
    static StepTotals collect(ProcessGroup group, long step, WeightedLoss loss, ParameterOptimizer optimizer) {
        float rate = optimizer instanceof AdamW adam ? adam.lr() : 0;
        byte[] local = Wire.encode(output -> {
            output.writeLong(step);
            output.writeLong(loss.weight());
            output.writeFloat(loss.sum().item());
            output.writeFloat(rate);
        });
        long weight = 0;
        double sum = 0;
        for (byte[] bytes : group.gather("step.totals", local)) {
            var contribution = decode(group, bytes, step, rate);
            weight = Math.addExact(weight, contribution.weight());
            sum += contribution.meanLoss();
        }
        if (weight == 0) RankAgreement.fail(group, "All data shards are exhausted");
        return new StepTotals(weight, (float) (sum / weight));
    }

    private static StepTotals decode(ProcessGroup group, byte[] bytes, long step, float rate) {
        try (var input = Wire.input(bytes)) {
            long observedStep = input.readLong(), weight = input.readLong();
            float loss = input.readFloat(), observedRate = input.readFloat();
            if (observedStep != step || weight < 0 || !Float.isFinite(loss) || observedRate != rate
                    || !Float.isFinite(rate) || input.read() != -1) {
                RankAgreement.fail(group, "Worker steps, losses, or learning rates differ");
            }
            return new StepTotals(weight, loss);
        } catch (IOException failure) { throw new CollectiveException("Invalid worker contribution", failure); }
    }
}
