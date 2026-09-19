package io.github.kirstenali.deepj.tensor;

public record CrossEntropyResult(Tensor loss, Tensor gradient) {

    public float meanLoss() {
        loss.materialize();
        return loss.data[0];
    }
}
