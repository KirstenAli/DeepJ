package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.autograd.Autograd;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.Trainable;

final class WorkerFault {
    private WorkerFault() {}

    static TrainingFixture inject(TrainingFixture fixture, String mode) {
        if (!mode.equals("nonfinite") && !mode.equals("backward-failure") && !mode.equals("gradient-nonfinite")) {
            return fixture;
        }
        TrainingObjective task = new TrainingObjective() {
            @Override public Trainable model() { return fixture.objective().model(); }
            @Override public WeightedLoss evaluate(int size) { return loss(fixture, mode); }
        };
        return new TrainingFixture(task, fixture.optimizer());
    }

    private static WeightedLoss loss(TrainingFixture fixture, String mode) {
        if (mode.equals("nonfinite")) return new WeightedLoss(Tensor.from(new float[]{Float.NaN}), 1);
        Tensor parameter = fixture.objective().model().parameters().get(0).value;
        Tensor output = Autograd.record(Tensor.from(new float[]{0}), gradient -> {
            if (mode.equals("backward-failure")) throw new IllegalStateException("Injected backward failure");
            return new Tensor[]{Tensor.backend().zeros(DType.FLOAT32, parameter.shape()).addScalar(Float.NaN)};
        }, parameter);
        return new WeightedLoss(output, 1);
    }
}
